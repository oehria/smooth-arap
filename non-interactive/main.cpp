#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <igl/read_triangle_mesh.h>
#include <igl/write_triangle_mesh.h>
#include <igl/writeOFF.h>
#include <igl/per_vertex_normals.h>
#include <igl/cotmatrix.h>
#include <igl/cotmatrix_entries.h>
#include <igl/adjacency_list.h>
#include <igl/barycenter.h>
#include <igl/massmatrix.h>
#include <igl/writeOBJ.h>
#include <igl/readOBJ.h>
#include <igl/slice.h>
#include <igl/slice_into.h>
#include <igl/invert_diag.h>
#include <igl/arap.h>
#include <igl/per_vertex_normals.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <cmath>
#include <igl/Timer.h>
#include<igl/massmatrix.h>

using namespace Eigen;
using namespace std;

Eigen::MatrixXd V, V1, C, LV0, LV1, rhs;
Eigen::MatrixXi F;
Eigen::SparseMatrix<double> L, A, M, M_inv;
VectorXd area;
std::vector<std::vector<int>> adj;
std::vector<int> B;
std::vector<Eigen::VectorXd> constrPositions;
string save_name;

enum Initialization { HANDLE, POISSON, BILAPLACIAN};
Initialization init = HANDLE;

int max_iters=2000;
float lambda;
const double convergence_thresh=0.0001;
int metric_iters;
double metric_time, metric_fac_time;
vector<double> metric_energy;
double prev_energy=2000;

//Constrained solver class
class ConstrainedLinearSolver {//solves linear system with some rows fixed (not unknown)
public:

    Eigen::MatrixXd xB;
    Eigen::VectorXi I, B;
    Eigen::MatrixXd b2;
    Eigen::Vector3i R3;

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> chol;

    //builds system
    ConstrainedLinearSolver(const Eigen::SparseMatrix<double>& A, const std::vector<int>& constr, const std::vector<Eigen::VectorXd>& values) {
        R3 << 0, 1, 2;

        // sort constraints
        std::vector<int> range;
        for (int i = 0; i < constr.size(); ++i) range.push_back(i);
        std::sort(range.begin(), range.end(), [&](const int i, const int j) {return constr[i] < constr[j]; });

        // fill constraint values, index free constraints
        if (!values.empty()) {
            xB.resize(values.size(), values[0].size());
            B.resize(constr.size());
        }

        int cnt = 0;
        for (int i : range) {
            B(cnt) = constr[i];
            xB.row(cnt) = values[i];
            ++cnt;
        }

        std::vector<int> rangeFull;
        for (int i = 0; i < A.cols(); ++i) rangeFull.push_back(i);
        I.resize(A.cols() - constr.size());
        std::set_difference(rangeFull.begin(), rangeFull.end(), B.data(), B.data() + B.size(), I.data());

        // build part matrices
        Eigen::SparseMatrix<double> AII, AIB;

        igl::slice(A, I, I, AII);
        chol.compute(AII);

        igl::slice(A, I, B, AIB);
        b2 = AIB * xB;
    }

    //solves system for given rhs b
    void solve(const Eigen::MatrixXd& b, Eigen::MatrixXd& x) {

        Eigen::MatrixXd bI;
        igl::slice(b, I, R3, bI);

        Eigen::MatrixXd xI = chol.solve(bI - b2);

        x.resizeLike(b);
        igl::slice_into(xB, B, R3, x);
        igl::slice_into(xI, I, R3, x);
    }
};


//spokes and rims mesh information data structure
struct Edge_sr {
    const int vertex_pos;//first vertex, positive sign
    const int vertex_neg;//second vertex, negative sign
    const int rid;//the rotation taken (some vertex number around)
    const double w;//the cotan (face) weight of this edge with this rid
};

//initializes spokes and rims datastructure used for later computations
//arguments: V holds original vertex positions, F faces, C cotan Laplace matrix and edgeSets_sr will hold resulting datastructure
void spokesAndRimsEdges_sr(Eigen::MatrixXd& V, 
    Eigen::MatrixXi& F, 
    Eigen::MatrixXd C,
    std::vector<std::vector<Edge_sr>>& edgeSets_sr) {

    const int nv = (int)V.rows();//number of vertices
    const int nf = (int)F.rows();//number of faces
    edgeSets_sr.resize(nv);//one entry per vertex because per-vertex rotations will be computed from this

    for (int i = 0; i < nf; i++) {//go over triangles
        for (int e = 0; e < 3; e++) {
            int j1 = (e+1)%3;//edge vertices indices
            int j2 = (e+2)%3;
            int vi = F(i, j1);//corresponding vertices
            int vj = F(i, j2);
            int vk = F(i, e);//opposite vertex (gives angle, identifies edge weight C(i,e))

            //push this edge with all 3 possible rotations and their corresponding weight into datastructure
            edgeSets_sr[vi].push_back({ vi, vj, vi, C(i,e)/3.0});//spoke rotation
            edgeSets_sr[vi].push_back({ vi, vj, vj, C(i,e)/3.0});//spoke rotation
            edgeSets_sr[vi].push_back({ vi, vj, vk, C(i,e)/3.0});//rim rotation
            //same for other vertex direction
            edgeSets_sr[vj].push_back({ vj, vi, vi, C(i,e)/3.0 });//spoke rotation
            edgeSets_sr[vj].push_back({ vj, vi, vj, C(i,e)/3.0 });//spoke rotation
            edgeSets_sr[vj].push_back({ vj, vi, vk, C(i,e)/3.0 });//rim rotation
        }
    }
}


//spokes and rims per vertex rotation fitting
//arguments: matrix V0 holding original vertex positions, V1 current vertex positions
//edgeSets_sr holds information about mesh structure, has to be initialized via spokesAndRimsEdges_sr
//rot is a vector of 3x3 matrices, which is where the rotation will be stored
void findRotations_sr(const Eigen::MatrixXd& V0,
    const Eigen::MatrixXd& V1,
    const std::vector<std::vector<Edge_sr>>& edgeSets_sr,
    std::vector<Eigen::Matrix3d>& rot) {

    const auto n = V0.rows();
    rot.clear();
    rot.resize(n, Eigen::Matrix3d::Zero());

    for (auto& edgeSet : edgeSets_sr) {
        for (auto& e : edgeSet) {
            const Eigen::Vector3d e0 = V0.row(e.vertex_pos) - V0.row(e.vertex_neg);
            const Eigen::Vector3d e1 = V1.row(e.vertex_pos) - V1.row(e.vertex_neg);
            rot[e.rid] += e.w * e0 * e1.transpose();
        }
    }

    // compute optimal rotations
    Eigen::Matrix3d flip = Eigen::Matrix3d::Identity();
    flip(2, 2) = -1.;
    for (int i = 0; i < n; ++i) {
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(rot[i], Eigen::ComputeFullU | Eigen::ComputeFullV);
        rot[i] = svd.matrixV() * svd.matrixU().transpose();
        if (rot[i].determinant() < 0) {
            rot[i] = svd.matrixV() * flip * svd.matrixU().transpose();
        }
    }
}

//builds rhs of equation to be solved, dependent on rotations
//arguments: V0 holds original vertex positions, LV0 original Laplacians
//eSets information about mesh structure, has to be initialized via spokesAndRimsEdges_sr
//L is cotan Laplace matrix, R holds rotations found via findRotations_sr
//lambda is smoothness parameter, rhs will hold the result of this function
void assembleRHS_sr(const Eigen::MatrixXd& V0,
    const Eigen::MatrixXd& LV0,
    const std::vector<std::vector<Edge_sr>>& eSets,
    const Eigen::SparseMatrix<double>& L,
    const std::vector<Eigen::Matrix3d>& R,
    const double lambda,
    Eigen::MatrixXd& rhs) {

    const auto nv = V0.rows();//number of vertices
    rhs.resize(nv, 3);
    rhs.setZero();

    if (1.0 - lambda) {
        Eigen::Vector3d b;
        int row = 0;
        for (auto& edgeSet : eSets) {//go over vertices
            b.setZero();
            for (auto& e : edgeSet) {//go over triangles/edges around vertex
                const Eigen::Vector3d e0 = V0.row(e.vertex_pos) - V0.row(e.vertex_neg);
                b += e.w * R[e.rid] * e0;//compute weighted, rotated edge
            }
            rhs.row(row) = (1.0 - lambda) *b;
            row++;
        }
    }
    if (lambda) {//quadratic/smooth term
        Eigen::MatrixXd b2;
        b2.resizeLike(LV0);
        b2.setZero();
        for (int i = 0; i < b2.rows(); ++i) {
            b2.row(i) = R[i] * LV0.row(i).transpose();//smooth arap has rotated lap vector
        }
        rhs += lambda *L *M_inv*b2;
    }
}


//scalar multiplication of sparse matrix
void mulSparse(Eigen::SparseMatrix<double>& L, const double f) {
    for (int i = 0; i < L.nonZeros(); ++i) L.valuePtr()[i] *= f;
}

//helper function: defines constraints for the survey experiments so they can be run simply via mesh name
void experiment(string mesh) {
    string path="../data/"+mesh;
    igl::read_triangle_mesh(path, V, F);
    std::cout << "mesh loaded " <<V.norm()<< endl;
    // set constraints
    if (mesh.compare("square_21_spikes.off")==0) {//spiked plane lifting
        for (int i = 0; i < 42; ++i) {
            B.push_back(i);
            Eigen::RowVector3d p = V.row(i);
            p(1) += 0.5;
            constrPositions.push_back(p);
        }

        for (int i = 399; i < V.rows(); ++i) {
            B.push_back(i);
            constrPositions.push_back(V.row(i));
        }
    }
    else if (mesh.compare("cactus.off")==0) {//survey cactus
        Matrix3d def_r;
        def_r << 0.999641, 0.021031, 0.016579,
            -0.022776, 0.342021, 0.939416,
            0.014087, -0.939457, 0.342378;
        Vector3d def_t;
        def_t << -0.019322, -0.134331, 0.821073;
        std::vector<int> fixed = { 200, 266, 276, 277, 288, 289, 290, 291, 292, 300, 304, 305, 306, 308, 314, 315, 331, 332, 348, 350, 353, 354, 356, 358, 473, 493, 496, 576, 577, 583, 584, 587, 588, 594, 595, 596, 601, 602, 603, 604, 607, 609, 611, 612, 613, 614, 615, 616, 617, 618, 619, 750, 758, 766, 767, 771, 774, 775, 779, 780, 788, 791, 798, 801, 802, 803, 844, 1136, 1139, 1531, 1585, 1586, 1587, 1630, 1631, 1648, 1649, 1650, 1682, 1684, 1691, 1692, 1693, 1694, 1695, 1706, 1707, 1708, 1709, 1710, 1711, 1740, 1741, 1742, 1743, 1750, 1751, 1752, 1753, 1754, 1755, 1777, 1778, 1782, 1783, 1784, 1785, 1786, 1790, 1804, 1805, 1806, 1807, 1808, 1809, 1811, 1812, 1813, 1816, 1819, 1820, 1821, 1822, 1823, 1824, 1825, 1826, 1827, 1828, 1829, 1830, 1831, 1832, 1833, 1834, 1835, 1836, 1837, 1839, 1840, 1841, 1842, 1843, 1844, 1845, 1846, 1847, 1848, 1849, 1850, 1851, 1852, 1853, 1854, 1855, 1965, 2052, 2172, 2173, 2174, 2186, 2187, 2201, 2202, 2203, 2205, 2213, 2214, 2215, 2222, 2223, 2224, 2230, 2233, 2244, 2267, 2268, 2269, 2274, 2276, 2277, 2278, 2279, 2280, 2281, 2302, 2329, 2335, 2369, 2419, 2578, 2830, 2848, 2852, 2860, 2885, 2903, 2910, 2914, 2916, 2924, 2935, 2939, 2941, 2942, 2943, 2959, 2971, 2979, 2980, 2981, 2994, 2995, 2996, 3001, 3003, 3014, 3026, 3027, 3028, 3029, 3030, 3031, 3037, 3038, 3039, 3040, 3041, 3042, 3043, 3060, 3061, 3062, 3065, 3066, 3072, 3073, 3074, 3076, 3077, 3080, 3081, 3082, 3083, 3084, 3086, 3087, 3088, 3089, 3091, 3092, 3093, 3094, 3095, 3096, 3097, 3098, 3101, 3102, 3103, 3104, 3107, 3108, 3109, 3111, 3112, 3113, 3114, 3115, 3116, 3117, 3118, 3119, 3120, 3121, 3122, 3124, 3125, 3126, 3127, 3128, 3130, 3172, 3437, 3480, 3481, 3489, 3535, 3536, 3538, 3540, 3588, 3593, 3594, 3724, 3725, 3726, 3727, 3728, 3729, 3736, 3753, 3754, 3755, 3756, 3757, 3758, 3760, 3761, 3776, 3777, 3778, 3779, 3793, 3795, 3796, 3797, 3801, 3820, 3852, 3853, 3858, 3859, 3860, 3864, 3865, 3866, 3867, 3868, 3869, 3870, 3871, 3872, 3873, 3874, 3875, 3876, 3897, 3919, 3925, 3956, 3965, 3969, 3979, 3981, 3985, 4047, 4169, 4182, 4201, 4250, 4280, 4507, 4710, 4834, 4836, 4837, 4838, 4904, 4905, 4906, 4907, 4946, 4965, 4966, 4967, 4969, 4970, 5005, 5006, 5007, 5016, 5017, 5018, 5019, 5020, 5032, 5033, 5034, 5035, 5036, 5037, 5038, 5060, 5070, 5071, 5080, 5102, 5104, 5105, 5109, 5110, 5113, 5114, 5115, 5120, 5121, 5132, 5134, 5135, 5136, 5138, 5140, 5141, 5144, 5145, 5146, 5147, 5148, 5149, 5150, 5151, 5152, 5153, 5154, 5155, 5156, 5157, 5158, 5159, 5160, 5161, 5162, 5164, 5165, 5166, 5167, 5168, 5169, 5170, 5171, 5172, 5173, 5174, 5175, 5176, 5177, 5179, 5180, 5182, 5183, 5184, 5186, 5187, 5188, 5190 };
        std::vector<int> handle = { 58, 84, 85, 95, 96, 104, 110, 128, 129, 130, 132, 151, 154, 155, 156, 177, 178, 181, 182, 183, 192, 194, 199, 203, 204, 218, 423, 424, 437, 445, 448, 460, 464, 465, 466, 467, 468, 469, 470, 481, 485, 498, 503, 504, 509, 516, 517, 525, 526, 529, 536, 542, 543, 568, 658, 662, 665, 682, 696, 706, 708, 714, 717, 719, 720, 726, 822, 939, 997, 998, 1055, 1069, 1070, 1112, 1130, 1133, 1134, 1135, 1137, 1138, 1140, 1141, 1142, 1144, 1151, 1152, 1153, 1194, 1195, 1197, 1198, 1216, 1217, 1218, 1219, 1220, 1223, 1225, 1235, 1236, 1275, 1281, 1282, 1291, 1292, 1293, 1294, 1297, 1298, 1308, 1309, 1341, 1342, 1343, 1344, 1345, 1346, 1349, 1352, 1355, 1356, 1372, 1373, 1374, 1375, 1376, 1378, 1379, 1393, 1426, 1427, 1428, 1429, 1430, 1431, 1436, 1450, 1451, 1453, 1454, 1455, 1457, 1458, 1495, 1496, 1497, 1498, 1515, 1516, 1555, 1556, 1712, 1954, 1970, 1971, 1972, 2043, 2044, 2070, 2071, 2076, 2077, 2078, 2091, 2092, 2093, 2100, 2101, 2102, 2106, 2107, 2108, 2119, 2121, 2252, 2282, 2434, 2438, 2447, 2448, 2453, 2488, 2501, 2502, 2503, 2504, 2505, 2506, 2552, 2553, 2554, 2555, 2556, 2569, 2570, 2587, 2595, 2613, 2616, 2617, 2618, 2619, 2620, 2621, 2629, 2630, 2631, 2632, 2633, 2634, 2636, 2637, 2662, 2663, 2664, 2665, 2666, 2679, 2680, 2681, 2682, 2683, 2691, 2692, 2693, 2694, 2695, 2726, 2727, 2731, 2732, 2733, 2754, 2755, 2756, 2772, 2778, 2779, 2796, 2797, 2800, 2819, 2836, 2855, 2856, 2857, 2886, 2887, 2888, 2889, 2890, 2906, 2907, 2972, 2993, 3216, 3236, 3242, 3254, 3258, 3259, 3263, 3319, 3320, 3322, 3323, 3324, 3352, 3353, 3356, 3424, 3470, 3471, 3472, 3478, 3479, 3518, 3519, 3520, 3521, 3522, 3523, 3529, 3530, 3531, 3532, 3533, 3534, 3556, 3557, 3558, 3559, 3560, 3573, 3574, 3575, 3576, 3577, 3578, 3585, 3586, 3587, 3589, 3590, 3591, 3592, 3611, 3612, 3613, 3614, 3615, 3616, 3780, 3803, 3839, 4155, 4156, 4157, 4158, 4243, 4244, 4264, 4323, 4324, 4347, 4348, 4349, 4350, 4351, 4352, 4353, 4355, 4356, 4369, 4370, 4371, 4372, 4373, 4388, 4412, 4424, 4425, 4426, 4427, 4428, 4429, 4437, 4438, 4451, 4452, 4456, 4472, 4473, 4519, 4535, 4536, 4544, 4546, 4547, 4548, 4549, 4551, 4555, 4556, 4557, 4558, 4559, 4560, 4561, 4565, 4581, 4582, 4583, 4593, 4620, 4621, 4622, 4624, 4625, 4626, 4635, 4636, 4657, 4658, 4659, 4663, 4664, 4680, 4722, 4723, 4724, 4725, 4726, 4727, 4735, 4736, 4751, 4752, 4753, 4781, 4795, 4803, 4804, 4832, 4875, 4876, 4903, 4960, 5002, 5044, 5079, 5089, 5106, 5128, 5143, 5230, 5231, 5237 };
        for (int i = 0; i < fixed.size(); i++) {
            B.push_back(fixed[i]);
            constrPositions.push_back(V.row(fixed[i]));
        }
        for (int i = 0; i < handle.size(); i++) {
            B.push_back(handle[i]);
            constrPositions.push_back(def_r * V.row(handle[i]).transpose() + def_t);
        }
    }
    else if (mesh.compare("bar.off") == 0) {//bar survey
        Matrix3d def_r;
        def_r << -0.707107, -0.707107, 0.0,
            0.707107, -0.707107, 0.0,
            0.0, 0.0, 1.0;
        std::vector<int> fixed = { 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 424, 425, 426, 427, 428, 429, 430, 431, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 949, 950, 951, 952, 953, 979, 980, 981, 982, 983, 984, 1010, 1011, 1012, 1013, 1014, 1015, 1042, 1043, 1044, 1045, 1046, 1268, 1269, 1270, 1271, 1272, 1273, 1274, 1275, 1290, 1291, 1292, 1293, 1294, 1295, 1296, 1297, 1298, 1299, 1300, 1301, 1302, 1303, 1312, 1313, 1314, 1315, 1316, 1317, 1318, 1319, 1320, 1321, 1322, 1323, 1324, 1325, 1326, 1327, 1328, 1329, 1330, 1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341, 1342, 1343, 1344, 1345, 1346, 1347, 1348, 1349, 1350, 1351, 1352, 1353, 1354, 1355, 1356, 1357, 1358, 1359, 1360, 1361, 1362, 1363, 1364, 1365, 1366, 1367, 1368, 1369, 1370, 1371, 1372, 1373, 1374, 1375, 1376, 1377, 1378, 1379, 2196, 2197, 2198, 2199, 2200, 2201, 2202, 2203, 2220, 2221, 2222, 2223, 2224, 2225, 2226, 2227, 2228, 2229, 2230, 2231, 2232, 2233, 2234, 2235, 2244, 2245, 2246, 2247, 2248, 2249, 2250, 2251, 2252, 2253, 2254, 2255, 2256, 2257, 2258, 2259, 2260, 2261, 2262, 2263, 2264, 2265, 2266, 2267, 2268, 2269, 2270, 2271, 2272, 2273, 2274, 2275, 2276, 2277, 2278, 2279, 2280, 2281, 2282, 2283, 2284, 2285, 2286, 2287, 2288, 2289, 2290, 2291, 2292, 2293, 2294, 2295, 2296, 2297, 2298, 2299, 2300, 2301, 2302, 2303, 2304, 2305, 2306, 2307, 2308, 2309, 2310, 2311, 2312, 2313, 2314, 2315, 3128, 3129, 3130, 3131, 3132, 3133, 3134, 3135, 3144, 3145, 3146, 3147, 3148, 3149, 3150, 3151, 3166, 3167, 3168, 3169, 3170, 3171, 3172, 3173, 3174, 3175, 3176, 3177, 3178, 3179, 3180, 3181, 3182, 3183, 3184, 3185, 3186, 3187, 3188, 3189, 3190, 3191, 3192, 3193, 3194, 3195, 3196, 3197, 3198, 3199, 3200, 3201, 3202, 3203, 3204, 3205, 3206, 3207, 3208, 3209, 3210, 3211, 3212, 3213, 3214, 3215, 3216, 3217, 3218, 3219, 3220, 3221, 3222, 3223, 3224, 3225, 3226, 3227, 3228, 3229, 3230, 3231, 3232, 3233, 3234, 3235, 3236, 3237, 3238, 3239 };
        std::vector <int> handle = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079, 1080, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 2893, 2894, 2895, 2896, 2897, 2898, 2899, 2900, 2901, 2902, 2903, 2904, 2905, 2906, 2907, 2908, 2909, 2910, 2911, 2912, 2913, 2914, 2915, 2916, 2917, 2918, 2919, 2920, 2921, 2922, 2923, 2924, 2925, 2926, 2927, 2928, 2929, 2930, 2931, 2932, 2933, 2934, 2935, 2936, 2937, 2938, 2939, 2940, 2941, 2942, 2943, 2944, 2945, 2946, 2947, 2948, 2949, 2950, 2951, 2952, 2953, 2954, 2955, 2956, 2957, 2958, 2959, 2960, 2961, 2962, 2963, 2964, 2965, 2966, 2967, 2968, 2969, 2970, 2971, 2972, 4384, 4385, 4386, 4387, 4388, 4389, 4390, 4391, 4392, 4393, 4394, 4395, 4396, 4397, 4398, 4399, 4400, 4401, 4402, 4403, 4404, 4405, 4406, 4407, 4408, 4409, 4410, 4411, 4412, 4413, 4414, 4415, 4416, 4417, 4418, 4419, 4420, 4421, 4422, 4423, 4424, 4425, 4426, 4427, 4428, 4429, 4430, 4431, 4432, 4433, 4434, 4435, 4436, 4437, 4438, 4439, 4440, 4441, 4442, 4443, 4444, 4445, 4446, 4447, 4448, 4449, 4450, 4451, 4452, 4453, 5294, 5295, 5296, 5297, 5298, 5299, 5300, 5301, 5302, 5303, 5304, 5305, 5306, 5307, 5308, 5309, 5310, 5311, 5312, 5313, 5314, 5315, 5316, 5317, 5318, 5319, 5320, 5321, 5322, 5323, 5324, 5325, 5326, 5327, 5328, 5329, 5330, 5331, 5332, 5333, 5334, 5335, 5336, 5337, 5338, 5339, 5340, 5341, 5342, 5343, 5344, 5345, 5346, 5347, 5348, 5349, 5350, 5351, 5352, 5353, 5354, 5355, 5356, 5357, 5358, 5359, 5360, 5361, 5362, 5363, 5364, 5365, 5366, 5367, 5368 };
        for (int i = 0; i < fixed.size(); i++) {
            B.push_back(fixed[i]);
            constrPositions.push_back(V.row(fixed[i]));
        }
        for (int i = 0; i < handle.size(); i++) {
            B.push_back(handle[i]);
            constrPositions.push_back(def_r * V.row(handle[i]).transpose());
        }
    }
    else if (mesh.compare("cylinder.off") == 0) {//suvey cylinder
        Matrix3d def_r;
        def_r << -0.5, 0.0, -0.866025,
            0.0, 1.0, 0.0,
            0.866025, 0.0, -0.5;
        Vector3d def_t;
        def_t << 5.196152, 0.0, 9.0;
        std::vector<int> fixed = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 4800 };
        std::vector <int> handle = { 4560, 4561, 4562, 4563, 4564, 4565, 4566, 4567, 4568, 4569, 4570, 4571, 4572, 4573, 4574, 4575, 4576, 4577, 4578, 4579, 4580, 4581, 4582, 4583, 4584, 4585, 4586, 4587, 4588, 4589, 4590, 4591, 4592, 4593, 4594, 4595, 4596, 4597, 4598, 4599, 4600, 4601, 4602, 4603, 4604, 4605, 4606, 4607, 4608, 4609, 4610, 4611, 4612, 4613, 4614, 4615, 4616, 4617, 4618, 4619, 4620, 4621, 4622, 4623, 4624, 4625, 4626, 4627, 4628, 4629, 4630, 4631, 4632, 4633, 4634, 4635, 4636, 4637, 4638, 4639, 4640, 4641, 4642, 4643, 4644, 4645, 4646, 4647, 4648, 4649, 4650, 4651, 4652, 4653, 4654, 4655, 4656, 4657, 4658, 4659, 4660, 4661, 4662, 4663, 4664, 4665, 4666, 4667, 4668, 4669, 4670, 4671, 4672, 4673, 4674, 4675, 4676, 4677, 4678, 4679, 4680, 4681, 4682, 4683, 4684, 4685, 4686, 4687, 4688, 4689, 4690, 4691, 4692, 4693, 4694, 4695, 4696, 4697, 4698, 4699, 4700, 4701, 4702, 4703, 4704, 4705, 4706, 4707, 4708, 4709, 4710, 4711, 4712, 4713, 4714, 4715, 4716, 4717, 4718, 4719, 4720, 4721, 4722, 4723, 4724, 4725, 4726, 4727, 4728, 4729, 4730, 4731, 4732, 4733, 4734, 4735, 4736, 4737, 4738, 4739, 4740, 4741, 4742, 4743, 4744, 4745, 4746, 4747, 4748, 4749, 4750, 4751, 4752, 4753, 4754, 4755, 4756, 4757, 4758, 4759, 4760, 4761, 4762, 4763, 4764, 4765, 4766, 4767, 4768, 4769, 4770, 4771, 4772, 4773, 4774, 4775, 4776, 4777, 4778, 4779, 4780, 4781, 4782, 4783, 4784, 4785, 4786, 4787, 4788, 4789, 4790, 4791, 4792, 4793, 4794, 4795, 4796, 4797, 4798, 4799, 4801 };

        for (int i = 0; i < fixed.size(); i++) {
            B.push_back(fixed[i]);
            constrPositions.push_back(V.row(fixed[i]));
        }
        for (int i = 0; i < handle.size(); i++) {
            B.push_back(handle[i]);
            constrPositions.push_back(def_r * V.row(handle[i]).transpose() + def_t);
        }
    }
    else if (mesh.compare("knubbel.off") == 0) {//knubbel survey
        Eigen::MatrixXd def = Eigen::MatrixXd::Zero(1, 3);
        def(0, 0) = -32.299956;//x
        def(0, 2) = 56.703263;//z
        //def(0, 0) = 100;//x
        double t = 3.14;
        Matrix3d def_r;
        def_r << cos(t), 0.0, -sin(t),
            0.0, 1, 0,
            sin(t), 0, cos(t);
        for (int i = 0; i < 1810; i++) {
            B.push_back(i);
            constrPositions.push_back(V.row(i));
        }
        for (int i = 38793; i < V.rows(); i++) {
            B.push_back(i);
            constrPositions.push_back((V.row(i).transpose()).transpose() +def);
        }
    }
    else{
        std::cout<<"ISSUE: UNKNOWN EXPERIMENT. Constraints are unclear, please use one of the survey experiments or specify your used constraints in the code yourself."<<std::endl;
    }
    //write handles (B) into text file
    // Create and open a text file
    ofstream MyFile("../handles.txt");
    // Write to the file
    for (int i = 0; i < B.size(); i++) {
        MyFile << B[i]<<" ";
    }
    // Close the file
    MyFile.close();
}

//naive bilaplacian initialization
void bilap_init(std::vector<std::vector<Edge_sr>> edgeSets_sr) {
    A = L * M_inv*L;
    ConstrainedLinearSolver lapsolver(A, B, constrPositions);//build constrained solver for system matrix
    std::vector<Eigen::Matrix3d> rot(V.rows(), Eigen::Matrix3d::Identity());//naive lap is identity
    assembleRHS_sr(V, LV0, edgeSets_sr, L, rot, 1.0, rhs);
    lapsolver.solve(rhs, V1);
}

//naive initialization via poisson system
void poi_init(std::vector<std::vector<Edge_sr>> edgeSets_sr) {
    A = L;
    ConstrainedLinearSolver lapsolver(A, B, constrPositions);//build constrained solver for system matrix
    std::vector<Eigen::Matrix3d> rot(V.rows(), Eigen::Matrix3d::Identity());//naive lap is identity
    assembleRHS_sr(V, LV0, edgeSets_sr, L, rot, 0, rhs);
    lapsolver.solve(rhs, V1);
}

//console input: mesh used [path to mesh file]
int main(int argc, const char* argv[]) {
    bool triangle = false;
    if (argc < 1) {
        cout << "ERROR: please provide path to mesh file" << endl;
    }
    experiment(argv[1]);

    //precomputations
    igl::cotmatrix(V, F, L);
    igl::cotmatrix_entries(V, F, C);
    std::vector<std::vector<Edge_sr>> edgeSets_sr;
    spokesAndRimsEdges_sr(V, F, C, edgeSets_sr);
    mulSparse(L, -1);
    LV0 = L * V;
    LV1 = LV0;
    igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_VORONOI, M);//write into mass matrix
    area=M.diagonal().eval();
    //idea: normalize s.t. each area entry is approx 1 (divide by sum, all sum to 1, times V which is number of areas)
    M/=area.sum();
    M*=V.rows();
    area/=area.sum();//normalize to area 1 so smoothness lambda not dependent on size of mesh
    area*=V.rows();
    igl::invert_diag(M, M_inv);//invert it
    //load constraints from experiment, this is just needed to display them later
    Eigen::MatrixXd bc(constrPositions.size(), 3);
    Eigen::MatrixXi b(B.size(), 1);
    for (int i = 0; i < bc.rows(); i++) {
        bc.row(i) = constrPositions[i];
        b(i) = B[i];
    }


    V1=V;
    //prepare viewer: color mesh, set normals, show points etc
    Eigen::MatrixXd col(V.rows(), 3);
    Eigen::RowVector3d point_color(102 / 255.0, 12 / 255.0, 33 / 255.0);
    Eigen::RowVector3d mesh_color(137/255.0, 200/255.0, 240/255.0);
    Eigen::RowVector3d back_color(1, 0, 0);
    for (int f = 0; f < V.rows(); f++)//go over vertices
    {
        col.row(f) = mesh_color;
    }
    igl::opengl::glfw::Viewer viewer;
    //add menu
    igl::opengl::glfw::imgui::ImGuiPlugin imgui_plugin;
    viewer.plugins.push_back(&imgui_plugin);
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    imgui_plugin.widgets.push_back(&menu);

    menu.callback_draw_viewer_menu = [&]()
	{
        if (ImGui::InputFloat("Smoothness Lambda [0,1]", &lambda, 0, 0));
        ImGui::InputText("File name", save_name);
        if (ImGui::Button("save .obj and stats", ImVec2(-1, 0)))
        {
            //save mesh
            std::fstream s{ "../res/" + save_name + ".obj", s.binary | s.trunc | s.in | s.out };
            for (int i = 0; i < V.rows(); i++) {
                s << "v " << V1(i, 0) << " " << V1(i, 1) << " " << V1(i, 2) << std::endl;
            }
            for (int i = 0; i < F.rows(); i++) {
                s << "f " << F(i, 0) + 1 << "/" << F(i, 0) + 1 << " "
                    << F(i, 1) + 1 << "/" << F(i, 1) + 1 << " "
                    << F(i, 2) + 1 << "/" << F(i, 2) + 1 << " " << std::endl;
            }
            s.close();
            //save metrics in .txt
            std::ofstream outFile("../res/" + save_name + ".txt");
            // Check if the file opened successfully
            if (!outFile) {
                std::cerr << "Failed to open the file for writing!" << std::endl;
            }
            // Write some text to the file
            outFile << "metrics for " << save_name << std::endl;
            outFile << metric_iters<<" iterations until convergence ("<<convergence_thresh<<")"<<endl;
            outFile<<"took "<<metric_time-metric_fac_time<<" s, excluding "<<metric_fac_time<<" s for factorization"<<endl;
            
            // Close the file
            outFile.close();
            std::cout << "Mesh and metrics have been saved!" << std::endl;

        }
        int init_type = static_cast<int>(init);
        if (ImGui::Combo("Initialization Scheme", &init_type, "Handle\0Poisson\0Bi-Laplacian\0"))
        {
            init = static_cast<Initialization>(init_type);
            if (init == HANDLE) {
                V1 = V;//move handles only
            }
            if (init == POISSON) {
                poi_init(edgeSets_sr);//solve poisson (L) equation
            }
            if (init == BILAPLACIAN) {
                bilap_init(edgeSets_sr);//solve bi-laplacian (L^2) equation -> naive laplacian editing
            }
            viewer.data().set_mesh(V1, F);//paint mesh. U is V, F faces

        }
        
        ImGui::InputInt("max", &max_iters);

        if (ImGui::Button("arap deformation", ImVec2(-1, 0))){
            //initialize here
            if (init == 0) {
                V1 = V;//move handles only
            }
            if (init == 1) {
                poi_init(edgeSets_sr);//solve poisson (L) equation
            }
            if (init == 2) {
                bilap_init(edgeSets_sr);//solve bi-laplacian (L^2) equation -> naive laplacian editing
            }
            std::vector<Eigen::Matrix3d> rot(V.rows(), Eigen::Matrix3d::Identity());
            igl::Timer time;
            time.start();
            //build system 
            A = lambda * L * M_inv* L + (1.0 - lambda) * L;
            ConstrainedLinearSolver solver(A, B, constrPositions);//build constrained solver for system matrix
            if (triangle){
                rot.resize(F.rows());
            }
            MatrixXd V_old=V;
            int iterations_ago = 100;
            bool converged=false;
            metric_fac_time=time.getElapsedTime();//save factorization time
            int i=0;//count iterations
            while(!converged&&i<max_iters){
                //local step
                findRotations_sr(V, V1, edgeSets_sr, rot);
                //build rhs for global step + solve global step
                assembleRHS_sr(V, LV0, edgeSets_sr, L, rot, lambda, rhs);
                solver.solve(rhs, V1);

                i++;
                if ((V1- V_old).norm()/V.norm() < convergence_thresh) {//small relative change --> convergence
					converged = true;
				}
				V_old = V1;//update
            }
            metric_time=time.getElapsedTime();
            metric_iters=i;
            viewer.data().set_mesh(V1, F);
        }
    };

    viewer.data().set_mesh(V1, F);
    //set shader normals
    Eigen::MatrixXd VN;
    igl::per_vertex_normals(V1, F, VN);
    // Set the viewer normals.
    viewer.data().set_normals(VN);
    viewer.data().set_colors(col);//paint color
    viewer.core().background_color.setOnes();
    viewer.data().point_size = 5;
    viewer.data().add_points(bc, point_color);//add points at constraints to see what was done
    viewer.data().show_lines = false;
    viewer.launch();

    return 0;
}

