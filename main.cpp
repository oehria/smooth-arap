//this is the new version

#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuizmoWidget.h>
#include <igl/opengl/glfw/imgui/SelectionWidget.h>
#include <GLFW/glfw3.h>
#include <igl/unproject_onto_mesh.h>
#include<igl/Timer.h>
#include<igl/cotmatrix.h>
#include<igl/massmatrix.h>
#include<igl/invert_diag.h>
#include <igl/cotmatrix_entries.h>
#include <igl/slice.h>
#include <igl/slice_into.h>
#include <igl/floor.h>
#include <igl/screen_space_selection.h>
#include <igl/adjacency_list.h>
#include <igl/readPLY.h>
#include <igl/arap.h>
#include <igl/screen_space_selection.h>
#include <igl/AABB.h>

//names
using namespace std;
using namespace Eigen;
using Viewer = igl::opengl::glfw::Viewer;

Eigen::MatrixXd V_orig, V_def, V;//vertex matrices
Eigen::MatrixXi F;//face matrix
Eigen::SparseMatrix<double> L, M, M_inv;//laplacian matrix, mass matrix
Eigen::MatrixXd Cov;//for rotation fitting
igl::opengl::glfw::Viewer viewer;
bool vertex_picking_mode = false;
bool handle_deleting_mode = false;
bool lap_rot = false;
bool naive = false;
bool no_mass = false;
int method = 1;//0 is spokes only
//1 spokes and rims
//2 triangle spokes and rims


std::vector<std::set<int>> areas;//each one contains itself and all others attached

igl::ARAPData arap_data;


//list of all vertices with their corresponding handle id. -1 if no handle
Eigen::VectorXi handle_id(0, 1);
//list of all vertices belonging to handles (id not -1), #HV x1
Eigen::VectorXi handle_vertices(0, 1);
//centroids of handle regions, #H x1
//Eigen::MatrixXd handle_centroids(0, 3);
//updated positions of handle vertices, #HV x3
Eigen::MatrixXd handle_vertex_positions(0, 3);
int num_handles = 0;
igl::opengl::glfw::imgui::ImGuizmoWidget guizmo;
MatrixXd pluginpos;
Eigen::Matrix4f T0 = guizmo.T;
igl::opengl::glfw::imgui::SelectionWidget selection;
igl::opengl::glfw::imgui::ImGuiMenu menu;
int plugin_vertex = 0;
Eigen::VectorXi v_free_index, v_constrained_index;
Eigen::SimplicialCholesky<SparseMatrix<double>> solver;
Eigen::SparseMatrix<double> free_influenced;
double lambda = 0.9;
std::vector<std::vector<int>> adj_list;
VectorXd sel_vertices;

Eigen::VectorXi cvi;//save how many triangles vertices are part of
struct Edge {
    const int i;
    const int j;
    const int rid;
    const double w;
};
std::vector<std::vector<Edge>> edgeSets;
std::vector<std::vector<Edge>> edgeSets_tr;

Eigen::RowVector3d point_color(102 / 255.0, 12 / 255.0, 33 / 255.0);
Eigen::RowVector3d mesh_color(137 / 255.0, 200 / 255.0, 240 / 255.0);
Eigen::MatrixXd VN;//vertex normals

//menu option stuff
enum Handle { LASSO, MARQUE, VERTEX, REMOVE, NONE};
Handle handle_option = NONE;
enum Trans { ROTATE, TRANSLATE, SCALE};
Trans transform_mode = TRANSLATE;
enum Method { SPOKES_ONLY,SPOKES_RIMS, TRIANGLE};
Method method_mode = SPOKES_RIMS;

//TRIANGLE SPOKES AND RIMS
void findRotations_triangles(const Eigen::MatrixXd& V0, const Eigen::MatrixXd& V1, const Eigen::MatrixXi& F, const Eigen::MatrixXd& C, const std::vector<std::vector<Edge>>& edgeSets, const Eigen::SparseMatrix<double>& L, const Eigen::MatrixXd& LV0, const double lambda, std::vector<Eigen::Matrix3d>& rot) {

    const auto n = V0.rows();//vertices
    const auto m = F.rows();//triangles

    rot.clear();
    rot.resize(m, Eigen::Matrix3d::Zero());

    for (int i = 0; i < m; i++) {
        Matrix3d temp, temp2;
        temp2.setZero();
        for (int j = 0; j < 2; ++j) {//go over TWO all edges surrounding faces
            const int k1 = j == 2 ? 0 : j + 1; //k1 is wrap-around successor of j
            const int k2 = k1 == 2 ? 0 : k1 + 1;//k2 is wrap-around successor of k1 (2nd to j)
            const Eigen::Vector3d e0 = V0.row(F(i, k1)) - V0.row(F(i, k2));//edge from face i k1 to k2 (goes over all edges due to j)
            const Eigen::Vector3d e1 = V1.row(F(i, k1)) - V1.row(F(i, k2));//same in current setting
            const Eigen::Matrix3d r = C(i, j) * e0 * e1.transpose();//weigh by cotangent weight//NOTE C(i,j)
            rot[i] += r;//add onto rotation matrix of triangle
            temp.row(j) = e0;
            temp2.row(j) = e1;
        }
        Eigen::Vector3d e00 = V0.row(F(i, 1)) - V0.row(F(i, 0));
        Eigen::Vector3d e01 = V0.row(F(i, 2)) - V0.row(F(i, 0));
        Eigen::Vector3d normal0 = (e00).cross(e01).normalized();
        temp.row(2) = normal0;
        rot[i] = temp.inverse() * temp2;
    }
    //  quadratic term
    if (lambda&&lap_rot) {
        const Eigen::MatrixXd LV1 = L * V1;

        for (int i = 0; i < n; ++i) {
            rot[i] += 2*lambda * LV0.row(i).transpose() * LV1.row(i);
        }
    }

    // compute optimal rotations
    Eigen::Matrix3d flip = Eigen::Matrix3d::Identity();
    flip(2, 2) = -1.;

    for (int i = 0; i < m; ++i) {

        Eigen::JacobiSVD<Eigen::Matrix3d> svd(rot[i], Eigen::ComputeFullU | Eigen::ComputeFullV);
        rot[i] = svd.matrixV() * svd.matrixU().transpose();

        if (rot[i].determinant() < 0) {
            rot[i] = svd.matrixV() * flip * svd.matrixU().transpose();
        }
    }
}

void trianglesEdges(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::SparseMatrix<double>& L, std::vector<std::vector<Edge>>& edgeSets) {
    const int nv = (int)L.rows();
    const int nf = (int)F.rows();
    edgeSets.resize(nv);

    Eigen::MatrixXd C;
    igl::cotmatrix_entries(V, F, C);

    cvi = Eigen::VectorXi::Zero(nv);

    /*
    * struct Edge {
    const int i;
    const int j;
    const int rid;
    const double w;
};
    */

    for (int i = 0; i < nf; ++i) {//go over triangles
        cvi(F(i, 0)) += 1;//count how many triangles this vertex is part of
        cvi(F(i, 1)) += 1;
        cvi(F(i, 2)) += 1;
        for (int j = 0; j < 3; ++j) {//for each vertex being in triangle

            int j0 = j;//vertex we focus on
            int j1 = (j0 == 2 ? 0 : j0 + 1);
            int j2 = (j1 == 2 ? 0 : j1 + 1);

            //int ijk[3]{ F(i, j0), F(i, j1), F(i, j2) };//put into data structure, will just be wrap around permuted

            //C(i,j) is weight of this triangle edge j1 j2
            //i will try to put: j1,j2 (edge), i (face id) and weight (C(i,j)) into my datastructure, for j1 and j2 as it only counts to it there 
            //std::cout << "WEIGHT " << C(i, j) << std::endl;
            edgeSets[F(i, j1)].push_back({ F(i,j1), F(i,j2), i, C(i,j) });
            edgeSets[F(i, j2)].push_back({ F(i,j2), F(i,j1), i, C(i,j) });
        }
    }
}

Eigen::MatrixXd rhs_triangles(const Eigen::MatrixXd& C,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& LV0,
    const std::vector<std::vector<Edge>>& eSets,
    const Eigen::SparseMatrix<double>& L,
    const std::vector<Eigen::Matrix3d>& R,
    const double lambda,
    Eigen::MatrixXd& rhs) {

    const auto nv = V.rows();
    rhs.resize(nv, 3);
    rhs.setZero();

    Eigen::Vector3d b;

    for (int i = 0; i < nv; ++i) {//go over vertices
        b.setZero();
        double check_sum = 0;
        for (auto& e : eSets[i]) {//go over triangles/edges around vertex
            b += e.w * R[e.rid] * (V.row(e.i) - V.row(e.j)).transpose();
            check_sum += e.w;
        }
        rhs.row(i) = (1.0 - lambda) * b;
    }

    if (lambda) {//lambda
        Eigen::MatrixXd b2;
        b2.resizeLike(LV0);
        b2.setZero();

        for (int i = 0; i < nv; ++i) {//go over vertices
            int count = 0;
            for (auto& e : eSets[i]) {//go over triangles/edges around vertex
                b2.row(i) += (2.0 / cvi(i)) * R[e.rid] * (LV0.row(i).transpose());
                count++;
            }
        }

        rhs += 0.25 * lambda * L * M_inv* b2;
    }
    return rhs;
}

void spokesAndRimsEdges_sr(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::SparseMatrix<double>& L, std::vector<std::vector<Edge>>& edgeSets_sr) {
    const int nv = (int)L.rows();
    const int nf = (int)F.rows();
    edgeSets_sr.resize(nv);//one entry per vertex because we then compute per vertex rotations from this

    Eigen::MatrixXd C;
    igl::cotmatrix_entries(V, F, C);
    cvi = Eigen::VectorXi::Zero(nv);

    /*
    struct Edge_sr {
        const int vertex_pos;//first vertex, positive sign
        const int vertex_neg;//second vertex, negative sign
        const int rid;//the rotation taken (some vertex number around)
        const double w;//the cotan (face) weight of this edge with this rid
    };
    */

    for (int i = 0; i < nf; i++) {//go over triangles
        //cvi(F(i, 0)) += 1;//count how many triangles this vertex is part of
        //cvi(F(i, 1)) += 1;
        //cvi(F(i, 2)) += 1;
        for (int e = 0; e < 3; e++) {//for each vertex being in triangle
            //j is opposite vertex (gives angle, identifies edge weight C(i,j))
            int j1 = (e + 1) % 3;//edge vertices
            int j2 = (e + 2) % 3;
            int vi = F(i, j1);
            int vj = F(i, j2);
            int vk = F(i, e);

            edgeSets_sr[vi].push_back({ vi, vj, vi, C(i,e) });//spoke rotation
            edgeSets_sr[vi].push_back({ vi, vj, vj, C(i,e) });//spoke rotation
            edgeSets_sr[vi].push_back({ vi, vj, vk, C(i,e) });//rim rotation

            edgeSets_sr[vj].push_back({ vj, vi, vi, C(i,e) });//spoke rotation
            edgeSets_sr[vj].push_back({ vj, vi, vj, C(i,e) });//spoke rotation
            edgeSets_sr[vj].push_back({ vj, vi, vk, C(i,e) });//rim rotation
        }
    }
}

Eigen::MatrixXd rhs_sr(const Eigen::MatrixXd& C,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& LV0,
    const std::vector<std::vector<Edge>>& eSets,
    const Eigen::SparseMatrix<double>& L,
    const std::vector<Eigen::Matrix3d>& R,
    const double lambda) {
    Eigen::MatrixXd rhs;
    const auto nv = V.rows();
    rhs.resize(nv, 3);
    rhs.setZero();
    Eigen::Vector3d b;
    int row = 0;
    for (auto& edgeSet : eSets) {//go over vertices
        b.setZero();
        for (auto& e : edgeSet) {//go over triangles/edges around vertex
            const Eigen::Vector3d e0 = V.row(e.i) - V.row(e.j);
            b += e.w / 3.0 * R[e.rid] * e0;//compute weighted, rotated edge
        }
        rhs.row(row) = (1.0 - lambda) * b;
        row++;
    }
    if (lambda) {//lambda
        Eigen::MatrixXd b2;
        b2.resizeLike(LV0);
        b2.setZero();

        for (int i = 0; i < nv; ++i) {//go over vertices
            b2.row(i) += 2* R[i] * (LV0.row(i).transpose());
        }

        rhs += 0.5 * lambda * L * M_inv * b2;
    }
    return rhs;
    //if (1.0 - lambda) {
    //    Eigen::Vector3d b;
    //    int row = 0;
    //    for (auto& edgeSet : eSets) {//go over vertices
    //        b.setZero();
    //        for (auto& e : edgeSet) {//go over triangles/edges around vertex
    //            const Eigen::Vector3d e0 = V0.row(e.vertex_pos) - V0.row(e.vertex_neg);
    //            b += e.w / 3.0 * R[e.rid] * e0;//compute weighted, rotated edge
    //        }
    //        rhs.row(row) = (1.0 - lambda) * b;
    //        row++;
    //    }
    //}
    //if (lambda) {//quadratic/smooth term
    //    Eigen::MatrixXd b2;
    //    b2.resizeLike(LV0);
    //    b2.setZero();
    //    for (int i = 0; i < b2.rows(); ++i) {
    //        b2.row(i) = R[i] * LV0.row(i).transpose();//smooth arap has rotated lap vector
    //    }
    //    rhs += lambda * L * b2;
    //}
}

void findRotations_sr(const Eigen::MatrixXd& V0,
    const Eigen::MatrixXd& V1,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& C,
    const std::vector<std::vector<Edge>>& edgeSets_sr,
    const Eigen::SparseMatrix<double>& L,
    const Eigen::MatrixXd& LV0,
    const double lambda,
    std::vector<Eigen::Matrix3d>& rot) {

    const auto n = V0.rows();
    rot.clear();
    rot.resize(n, Eigen::Matrix3d::Zero());

    for (auto& edgeSet : edgeSets_sr) {
        for (auto& e : edgeSet) {
            const Eigen::Vector3d e0 = V0.row(e.i) - V0.row(e.j);
            const Eigen::Vector3d e1 = V1.row(e.i) - V1.row(e.j);
            if (lap_rot) {
                rot[e.rid] += e.w * (1 - lambda) * e0 * e1.transpose();
            }
            else {
                rot[e.rid] += e.w * e0 * e1.transpose();//lamda irrelevant, allows to do regularization experiment
            }
        }
    }
    //  quadratic term
    if (lap_rot && lambda) {
        const Eigen::MatrixXd LV1 = L * V1;

        for (int i = 0; i < n; ++i) {
            rot[i] += 2*lambda *M_inv.coeff(i, i) * LV0.row(i).transpose() * LV1.row(i);
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
        if (naive) {
            rot[i].setIdentity();
        }
    }
}

void factorize(Viewer& viewer, double lambda) {
    if (num_handles == 0) {
        return;
    }
    Eigen::SparseMatrix<double> bi_lap, solver_mat;

    igl::massmatrix(V_orig, F, igl::MASSMATRIX_TYPE_VORONOI, M);//write into mass matrix
    igl::invert_diag(M, M_inv);//invert it
    double meanminv = 0;//mean of the inverted mass matrix
    for (int i = 0; i < M_inv.rows(); i++) {
        meanminv += M_inv.coeff(i, i) / M_inv.rows();
    }
    M_inv = M_inv / meanminv;//normalize 
    if (no_mass) {
        M_inv.setIdentity();//NO MASS USED
    }
    //construct system matrix 
    bi_lap = lambda * L * M_inv * L - (1.0 - lambda) * L;
    //constrain system
    int num_free = V_orig.rows() - handle_vertices.size();
    v_free_index.resize(num_free);
    v_constrained_index = handle_vertices;
    int count_free = 0;
    for (int i = 0; i < handle_id.size(); ++i) {
        if (handle_id[i] == -1) {
            v_free_index[count_free++] = i;
        }
    }
    igl::slice(bi_lap, v_free_index, v_free_index, solver_mat);
    igl::slice(bi_lap, v_free_index, v_constrained_index, free_influenced);
    solver.compute(solver_mat);
    //libigl arap for debugging
#if 0
    arap_data.max_iter = 1;
    //chao:elements
    arap_data.energy = igl::ARAP_ENERGY_TYPE_SPOKES;
    igl::arap_precomputation(V_orig, F, V.cols(), handle_vertices, arap_data);//nonsmooth precomp
#endif
}

void findRotations_spokes(const Eigen::MatrixXd& V0,
    const Eigen::MatrixXd& V1,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& C,
    std::vector<Eigen::Matrix3d>& rot) {
    const auto n = V0.rows();
    const auto m = F.rows();
    rot.clear();
    rot.resize(n, Eigen::Matrix3d::Zero());

    // regular term
    // edge rotation fitting
    for (int i = 0; i < m; ++i) {//go over faces
        for (int j = 0; j < 3; ++j) {//go over all edges surrounding faces
            const int k1 = (j + 1) % 3; //k1 is wrap-around successor of j
            const int k2 = (j + 2) % 3;//k2 is wrap-around successor of k1 (2nd to j)
            const Eigen::Vector3d e0 = V0.row(F(i, k1)) - V0.row(F(i, k2));//edge from face i k1 to k2 (goes over all edges due to j)
            const Eigen::Vector3d e1 = V1.row(F(i, k1)) - V1.row(F(i, k2));//same in current setting
            Eigen::Matrix3d r;
            if (lap_rot) {
                r = C(i, j) * (1 - lambda) * e0 * e1.transpose();//weigh by cotangent weight of face
            }
            else {
                r = C(i, j) * e0 * e1.transpose();//weigh by cotangent weight of face
            }

            //spokes only
            rot[F(i, k1)] += r;//add onto rotation matrix of the two involved vertices
            rot[F(i, k2)] += r;
        }
    }
    //  quadratic term
    if (lambda && lap_rot) {
        const Eigen::MatrixXd LV1 = L * V1;
        const Eigen::MatrixXd LV0 = L * V0;

        for (int i = 0; i < n; ++i) {
            rot[i] += 2*lambda *M_inv.coeff(i,i)* LV0.row(i).transpose() * LV1.row(i);
        }
    }
    // compute optimal rotations
    Eigen::Matrix3d flip = Eigen::Matrix3d::Identity();//use to flip last row in case it's reflection and not rotation
    flip(2, 2) = -1.;
    for (int i = 0; i < n; ++i) {//for every vertex
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(rot[i], Eigen::ComputeFullU | Eigen::ComputeFullV);//compute svd
        rot[i] = svd.matrixV() * svd.matrixU().transpose();//compute rotation
        if (rot[i].determinant() < 0) {//det might be -1, in which case it is reflection -> flip so we have rotation
            rot[i] = svd.matrixV() * flip * svd.matrixU().transpose();
        }
    }
}

Eigen::MatrixXd rhs_spokes(Viewer& viewer, double lambda, const std::vector<Eigen::Matrix3d>& R) {
    Eigen::MatrixXd b2;
    Eigen::MatrixXd LV0 = L * V_orig;
    b2.resizeLike(LV0);
    Eigen::MatrixXd rhs;
    rhs.resizeLike(b2);
    rhs.setZero();

    for (int i = 0; i < F.rows(); ++i) {//go over faces
        for (int j = 0; j < 3; ++j) {//go over edges surrounding face
            int v0 = F(i, (j + 1) % 3);//same as k1
            int v1 = F(i, (j + 2) % 3);//same as k2
            Eigen::Vector3d b;
            b = 0.5 * (1.0 - lambda) * Cov(i, j) * (R[v0] + R[v1]) * (V_orig.row(v0) - V_orig.row(v1)).transpose();//original arap rhs formula
            rhs.row(v0) += b.transpose();
            rhs.row(v1) -= b.transpose();
        }
    }

    for (int i = 0; i < b2.rows(); ++i) {
        b2.row(i) = 2*R[i] * LV0.row(i).transpose();//smooth arap has rotated lap vector
    }
    rhs += 0.5 * lambda * L * M_inv * b2;
    return rhs;
}

bool solve(Viewer& viewer) {
    if (num_handles == 0) {
        return false;
    }
    Eigen::VectorXi R3(3);
    R3 << 0, 1, 2;
    //igl::slice_into(handle_vertex_positions, handle_vertices, R3, V);
    //viewer.data().set_vertices(V);

    Eigen::MatrixXd b2 = free_influenced * handle_vertex_positions;//constraints
    //rotation fitting
    std::vector<Eigen::Matrix3d> R;
    Eigen::MatrixXd b;
    if (method == 0) {//spokes only arap
        findRotations_spokes(V_orig, V, F, Cov, R);
        //system rhs (depending on rotation)
        b = rhs_spokes(viewer, lambda, R);//arap rhs
    }
    if (method == 1) {//spokes and rims arap
        findRotations_sr(V_orig, V, F, Cov, edgeSets, L, L * V_orig, lambda, R);
        b = rhs_sr(Cov, V_orig, F, L * V_orig, edgeSets, L, R, lambda);
    }
    if (method == 2) {//spokes and rims arap
        findRotations_triangles(V_orig, V, F, Cov, edgeSets_tr, L, L * V_orig, lambda, R);
        rhs_triangles(Cov, V_orig, F, L * V_orig, edgeSets_tr, L, R, lambda, b);
    }
    Eigen::MatrixXd bI;
    igl::slice(b, v_free_index, R3, bI);//only non-handle part of arap rhs
    Eigen::MatrixXd V_free_deformed = solver.solve(bI - b2);//solve constrained

    V_def = V_orig;
    igl::slice_into(V_free_deformed, v_free_index, R3, V_def);
    igl::slice_into(handle_vertex_positions, handle_vertices, R3, V_def);
    V = V_def;
    viewer.data().set_vertices(V);
    viewer.data().compute_normals();
    viewer.data().clear_points();
    viewer.data().add_points(handle_vertex_positions, point_color);
    return true;
};

void compute_handle_centroid(MatrixXd& handle_centroid, int id)
{
    //compute centroid of handle
    handle_centroid.setZero(1, 3);
    int num = 0;
    for (long vi = 0; vi < V.rows(); ++vi)
    {
        int r = handle_id[vi];
        if (r == id)
        {
            handle_centroid += V.row(vi);
            num++;
        }
    }

    handle_centroid = handle_centroid.array() / num;

}

bool make_area_handle() {
    vector<int> area_handle_vertices;
    set<int> contained_handles;
    for (int i = 0; i < sel_vertices.size(); i++) {
        if (sel_vertices(i) > 0.9) {
            area_handle_vertices.push_back(i);//vertex index i is part of this handle
            if (handle_id(i) != -1) {
                contained_handles.insert(i);
            }
        }
    }
    int id = area_handle_vertices[0];
    int old_num_handles = num_handles;
    num_handles += area_handle_vertices.size() - contained_handles.size();
    Eigen::VectorXi up_handle_vertices = Eigen::VectorXi(num_handles);//update handle vertex vector
    Eigen::MatrixXd up_handle_pos = Eigen::MatrixXd(num_handles, 3);//update handle vertex vector
    int count = 0;
    for (int i = 0; i < old_num_handles; i++) {
        if (contained_handles.count(handle_vertices(i)) == 0) {
            up_handle_vertices(count) = handle_vertices(i);
            up_handle_pos.row(count) = handle_vertex_positions.row(i);
            count++;
        }
    }
    for (int i = 0; i < area_handle_vertices.size(); i++) {
        handle_id(area_handle_vertices[i]) = id;
        up_handle_vertices(count + i) = area_handle_vertices[i];
        up_handle_pos.row(count + i) = V.row(area_handle_vertices[i]);
    }
    handle_vertices = up_handle_vertices;
    handle_vertex_positions = up_handle_pos;
    plugin_vertex = area_handle_vertices[0];
    viewer.data().clear_points();
    viewer.data().add_points(handle_vertex_positions, point_color);
    compute_handle_centroid(pluginpos, id);
    guizmo.T.block(0, 3, 3, 1) = pluginpos.transpose().cast<float>();
    guizmo.visible = true;
    factorize(viewer, lambda);
    selection.mode = igl::opengl::glfw::imgui::SelectionWidget::OFF;
    handle_option = NONE;
    return true;
}

int main(int argc, char* argv[]) {
    // Load a mesh from file
    std::string filename;
    if (argc == 2) {
        filename = "C:/Users/annik/source/repos/oehria-code/libigl/tutorial/109_ImGuizmo/data/"+std::string(argv[1]); // Mesh provided as command line argument
    }
    else {
        filename = std::string("C:/Users/annik/source/repos/oehria-code/libigl/tutorial/109_ImGuizmo/data/square_21_spikes.off"); // Default mesh
    }
    igl::read_triangle_mesh(filename, V, F);
    V.resize(5, 3);
    V.row(0) << -1, -1, 0;
    V.row(1) << 0, -1, 0;
    V.row(2) << 1, -1, 0;
    V.row(3) << 0, 1, 0;
    V.row(4) << 1, 1, 0;
    F.resize(3, 3);
    F.row(0) << 0, 1, 3;
    F.row(1) << 1, 2, 3;
    F.row(2) << 4, 3, 2;
    V_orig = V;
    viewer.data().set_mesh(V, F);

    //precomputations on mesh
    igl::adjacency_list(F, adj_list);
    igl::cotmatrix_entries(V_orig, F, Cov);//for rotation 
    igl::cotmatrix(V_orig, F, L);//write into global L
    spokesAndRimsEdges_sr(V_orig, F, L, edgeSets);
    trianglesEdges(V_orig, F, L, edgeSets_tr);
    igl::AABB<Eigen::MatrixXd, 3> tree;
    tree.init(V, F);
    Eigen::Array<double, Eigen::Dynamic, 1> and_visible =
        Eigen::Array<double, Eigen::Dynamic, 1>::Zero(V.rows());

    //initialize handles 
    //id to -1 because nothing was assigned yet
    handle_id.setConstant(V.rows(), 1, -1);
    //handle plugins for deformation, selection, menu
    igl::opengl::glfw::imgui::ImGuiPlugin imgui_plugin;
    viewer.plugins.push_back(&imgui_plugin);
    // Add a 3D gizmo plugin
    guizmo.operation = ImGuizmo::TRANSLATE;
    imgui_plugin.widgets.push_back(&guizmo);
    guizmo.visible = false;
    guizmo.T.block(0, 3, 3, 1) = V.row(plugin_vertex).transpose().cast<float>();
    //Add selection plugin
    selection.mode = igl::opengl::glfw::imgui::SelectionWidget::OFF;
    imgui_plugin.widgets.push_back(&selection);
    //add menu
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    imgui_plugin.widgets.push_back(&menu);

    // Attach callback to apply imguizmo's transform to mesh
    guizmo.callback = [&](const Eigen::Matrix4f& T)
    {
        T0.block(0, 3, 3, 1) = pluginpos.transpose().cast<float>();
        const Eigen::Matrix4d TT = (T * T0.inverse()).cast<double>().transpose();
        V_def = (V.rowwise().homogeneous() * TT).rowwise().hnormalized();
        int id = handle_id(plugin_vertex);
        if (handle_vertex_positions.rows() > 0) {
            for (int i = 0; i < handle_vertices.rows(); i++) {//EFFICIENCY make list of vertices belonging to this handle instead
                if (handle_id(handle_vertices(i)) == id) {//this handle
                    handle_vertex_positions.row(i) = V_def.row(handle_vertices(i));//update handle pos
                }
            }
            solve(viewer);
        }
        T0 = T;
        pluginpos = ((pluginpos.rowwise().homogeneous() * TT).rowwise().hnormalized()).eval();
    };

    //selection plugin
    selection.callback = [&]()
    {
        igl::screen_space_selection(V, F, tree, viewer.core().view, viewer.core().proj, viewer.core().viewport, selection.L, sel_vertices, and_visible);
        make_area_handle();
    };

    //draw menu
    menu.callback_draw_viewer_menu = [&]()
    {
        // Draw parent menu content
        //menu.draw_viewer_menu();
        // Add new group
        if (ImGui::CollapsingHeader("Deformation Controls", ImGuiTreeNodeFlags_DefaultOpen))
        {
            int trans_type = static_cast<int>(transform_mode);
            if (ImGui::Combo("Transformation Mode", &trans_type, "ROTATE\0TRANSLATE\0SCALE\0"))
            {
                transform_mode = static_cast<Trans>(trans_type);
                if (transform_mode == TRANSLATE) {
                    viewer.callback_key_pressed(viewer, 't', 0);
                }
                if (transform_mode == ROTATE) {
                    viewer.callback_key_pressed(viewer, 'r', 0);
                }
                if (transform_mode == SCALE) {
                    guizmo.operation = ImGuizmo::SCALE;
                }
            }
            int handle_type = static_cast<int>(handle_option);
            if (ImGui::Combo("Handle Option", &handle_type, "LASSO\0MARQUE\0VERTEX\0REMOVE\0NONE\0"))
            {
                handle_option = static_cast<Handle>(handle_type);
                if (handle_option == LASSO) {
                    selection.mode = igl::opengl::glfw::imgui::SelectionWidget::LASSO;
                    viewer.callback_key_pressed(viewer, 'l', 0);
                }
                if (handle_option == MARQUE) {
                    selection.mode = igl::opengl::glfw::imgui::SelectionWidget::RECTANGULAR_MARQUEE;
                    viewer.callback_key_pressed(viewer, 'm', 0);
                }
                if (handle_option == VERTEX) {
                    viewer.callback_key_pressed(viewer, 'p', 0);
                }
                if (handle_option == NONE) {
                    viewer.callback_key_pressed(viewer, 'v', 0);
                }
                if (handle_option == REMOVE) {
                    viewer.callback_key_pressed(viewer, 'x', 0);
                }
            }
            if (ImGui::Combo("Deformation Method", &method, "SPOKES_ONLY\0SPOKES_RIMS\0TRIANGLES\0"))
            {
                method_mode = static_cast<Method>(method);
                factorize(viewer, lambda); 
                solve(viewer);
            }
            if (ImGui::InputDouble("Smoothness lambda [0,1]", &lambda, 0, 0)) {
                factorize(viewer, lambda); 
                solve(viewer);
            }
            ImGui::Checkbox("Full rotation fitting", &lap_rot);
            if (ImGui::Checkbox("No rotation (naive Laplacian)", &naive)) {
                factorize(viewer, lambda);
                solve(viewer);
            };
            if (ImGui::Checkbox("No Mass", &no_mass)) {
                factorize(viewer, lambda);
                solve(viewer);
            };
            if (ImGui::Button("10 iterations", ImVec2(-1, 0)))
            {
                viewer.callback_key_pressed(viewer, 'c', 0);
            }
        }
    };

    // Maya-style keyboard shortcuts for operation
    viewer.callback_key_pressed = [&](decltype(viewer)&, unsigned int key, int mod)
    {
        cout << "keyy" << endl;
        vertex_picking_mode = false;
        handle_deleting_mode = false;
        switch (key)
        {
        case 'T': case 't': guizmo.operation = ImGuizmo::TRANSLATE;  transform_mode = TRANSLATE;  return true;
        case 'R': case 'r': guizmo.operation = ImGuizmo::ROTATE;  transform_mode = ROTATE;  return true;
       /* case 'V': case 'v': vertex_picking_mode = false; handle_option = NONE;  return true;
        case 'M': case 'm': cout << "mmm" << endl; handle_option = MARQUE;  return true;
        case 'l': handle_option = LASSO;  return true;*/
        case 'P': case 'p': vertex_picking_mode = true; handle_option = VERTEX;  selection.mode = igl::opengl::glfw::imgui::SelectionWidget::OFF; return true;//try to add vertex picking mode 
        case 'X': case 'x': handle_deleting_mode = true; handle_option = REMOVE; return true;
        case 'C': case 'c':
            for (int i = 0; i < 10; i++) {
                solve(viewer);
            }
            return true;
        }
        return false;
    };

    //if vertex picking mode will add handle
    viewer.callback_mouse_down =[&](igl::opengl::glfw::Viewer& viewer, int, int)->bool
    {
        if (true) {
            int fid;
            Eigen::Vector3f bc;
            // Cast a ray in the view direction starting from the mouse position
            double x = viewer.current_mouse_x;
            double y = viewer.core().viewport(3) - viewer.current_mouse_y;
            if (igl::unproject_onto_mesh(Eigen::Vector2f(x, y), viewer.core().view,
                viewer.core().proj, viewer.core().viewport, V, F, fid, bc))
            {
                if (!(vertex_picking_mode || handle_deleting_mode)) {
                    return false;
                }
                float max = bc.maxCoeff();//point that is closest
                int point_face_idx = 0;//find argmax
                for (int i = 0; i < 3; i++) {
                    if (bc(i) == max) {
                        point_face_idx = i;
                    }
                }
                int point = F(fid, point_face_idx);//indexes into V
                if (!handle_deleting_mode) { //now add that vertex to handle
                    //check if already exists
                    if (handle_id(point) != -1) {
                        plugin_vertex = point;
                        guizmo.visible = true;
                        compute_handle_centroid(pluginpos, handle_id(point));
                        guizmo.T.block(0, 3, 3, 1) = pluginpos.transpose().cast<float>();//V.row(plugin_vertex).transpose().cast<float>();//position
                        return true;
                    }
                    //is new vertex
                    Eigen::VectorXd pt = V.row(point);
                    viewer.data().add_points(V.row(point), point_color);
                    num_handles++;
                    //plugin_vertex_index = num_handles - 1;
                    Eigen::VectorXi up_handle_vertices = Eigen::VectorXi(num_handles);//update handle vertex vector
                    Eigen::MatrixXd up_handle_pos = Eigen::MatrixXd(num_handles, 3);//update handle vertex vector
                    for (int i = 0; i < num_handles - 1; i++) {
                        up_handle_vertices(i) = handle_vertices(i);
                        up_handle_pos.row(i) = handle_vertex_positions.row(i);
                    }
                    up_handle_vertices(num_handles - 1) = point;
                    up_handle_pos.row(num_handles - 1) = V.row(point);
                    handle_vertices = up_handle_vertices;
                    handle_vertex_positions = up_handle_pos;
                    plugin_vertex = point;
                    handle_id(plugin_vertex) = plugin_vertex;//set id to itself
                    factorize(viewer, lambda);
                    //now make plugin 'active' at that location
                    guizmo.visible = true;
                    pluginpos = V.row(point);
                    guizmo.T.block(0, 3, 3, 1) = pluginpos.transpose().cast<float>();//position
                }
                else {
                    vector<int> handle_to_delete_idx;
                    bool found = false;
                    for (int i = 0; i < num_handles; i++) {
                        if (handle_id(handle_vertices(i)) == handle_id(point)) {//check if selected handle is at index i in the handle vector
                            handle_to_delete_idx.push_back(i);
                            found = true;
                        }
                    }
                    if (!found) {
                        cout << "no handle to delete" << endl;
                        return false;
                    }
                    else {
                        for (int i = 0; i < handle_to_delete_idx.size(); i++) {
                            handle_id(handle_vertices(handle_to_delete_idx[i])) = -1;//does not belong to handle anymore
                        }
                        num_handles -= handle_to_delete_idx.size();
                        Eigen::VectorXi up_handle_vertices = Eigen::VectorXi(num_handles);//update handle vertex vector
                        Eigen::MatrixXd up_handle_pos = Eigen::MatrixXd(num_handles, 3);//update handle vertex vector
                        int curr = 0;
                        for (int i = 0; i < num_handles + handle_to_delete_idx.size(); i++) {
                            if (handle_id(handle_vertices(i)) != handle_id(point)) {
                                up_handle_vertices(curr) = handle_vertices(i);
                                up_handle_pos.row(curr) = handle_vertex_positions.row(i);
                                curr++;
                            }
                        }
                        handle_vertices = up_handle_vertices;
                        handle_vertex_positions = up_handle_pos;
                        if (num_handles > 0) {
                            factorize(viewer, lambda);
                            solve(viewer);
                        }
                        else {
                            V = V_orig;
                            viewer.data().set_vertices(V);
                        }
                    }
                    viewer.data().clear_points();
                    viewer.data().add_points(handle_vertex_positions, point_color);
                }

                return true;
            }
            return false;
        }
    };


    //display shortcut keys
    std::cout << R"(
T,t   Switch to translate operation
R,r   Switch to rotate operation
P, p  Click to select a vertex
X, x  Click on handle to remove
M, m  Area marquee selection
l     Area lasso selection
v     Switch off handle selection
)";

    //set up viewer
    Eigen::MatrixXd col(V.rows(), 3);
    for (int i = 0; i < V.rows(); i++)//go over vertices
    {
        col.row(i) = mesh_color;
    }
    viewer.data().set_colors(col);//paint color
    viewer.data().compute_normals();
    viewer.data().show_lines = false;
    viewer.data().point_size = 10;
    viewer.core().background_color.setOnes();
    viewer.launch();
}