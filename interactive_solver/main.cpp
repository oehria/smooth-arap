#include <memory>

#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuizmoWidget.h>
#include <igl/opengl/glfw/imgui/SelectionWidget.h>
#include <GLFW/glfw3.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/Timer.h>
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <igl/invert_diag.h>
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
#include <igl/stb/read_image.h>


#include "ConstrainedLinearSolver.hpp"


//names
using namespace std;
using namespace Eigen;
using Viewer = igl::opengl::glfw::Viewer;

//user parameter - select lambda :)
double lambda = 0.995;

Eigen::MatrixXd V_orig, V_def, V;//vertex matrices
Eigen::MatrixXi F;//face matrix
Eigen::SparseMatrix<double> L, L_sr, M, M_inv;//laplacian matrix, mass matrix
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
string save_name;

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
std::unique_ptr<ConstrainedLinearSolver> constrainedSolver;
Eigen::SparseMatrix<double> free_influenced;
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

Eigen::RowVector3d point_color(1, 0.85, 0.40);
Eigen::RowVector3d mesh_color(137 / 255.0, 200 / 255.0, 240 / 255.0);
Eigen::MatrixXd VN;//vertex normals

//menu option stuff
enum Handle { VERTEX, REMOVE, NONE};
Handle handle_option = NONE;
enum Trans { TRANSLATE};
Trans transform_mode = TRANSLATE;

//ORIGINAL (SPOKES AND RIMS) ARAP 
void spokesAndRimsEdges_sr(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::SparseMatrix<double>& L, std::vector<std::vector<Edge>>& edgeSets_sr) {//TODO: adjust weight
    const int nv = (int)L.rows();
    const int nf = (int)F.rows();
    edgeSets_sr.resize(nv);//one entry per vertex because we then compute per vertex rotations from this

    Eigen::MatrixXd C;
    igl::cotmatrix_entries(V, F, C);
    cvi = Eigen::VectorXi::Zero(nv);


    for (int i = 0; i < nf; i++) {//go over triangles
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
        if (naive) {
            rot[i].setIdentity();
        }
    }
}

void factorize(Viewer& viewer, double lambda) {
    
    Eigen::SparseMatrix<double> bi_lap, solver_mat;

    igl::massmatrix(V_orig, F, igl::MASSMATRIX_TYPE_VORONOI, M);//write into mass matrix
    igl::invert_diag(M, M_inv);//invert it
    double meanminv = 0;//mean of the inverted mass matrix
    for (int i = 0; i < M_inv.rows(); i++) {
        meanminv += M_inv.coeff(i, i) / M_inv.rows();
    }
    M_inv = M_inv / meanminv;//normalize 

    //construct system matrix
    Eigen::SparseMatrix<double> Id(L.cols(), L.cols());
    Id.setIdentity();
    
    bi_lap = lambda * L * M_inv * L - (1.0 - lambda) * L;
    
    constrainedSolver.reset(new ConstrainedLinearSolver(bi_lap));//no fixed
    return;
}

bool solve(Viewer& viewer) {
    if (num_handles == 0) {
        return false;
    }
    
    Eigen::VectorXi R3(3);
    R3 << 0, 1, 2;

    //rotation fitting
    std::vector<Eigen::Matrix3d> R;
    Eigen::MatrixXd b;
    //spokes and rims arap
    findRotations_sr(V_orig, V, F, Cov, edgeSets, L, L * V_orig, lambda, R);
    b = rhs_sr(Cov, V_orig, F, L * V_orig, edgeSets, L, R, lambda);
    constrainedSolver->solve(b, handle_vertex_positions, V_def);
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
        filename = "../data/"+std::string(argv[1]); // Mesh provided as command line argument
    }
    else {
        filename = std::string("../data/dog.obj"); // Default mesh
    }
    
    igl::read_triangle_mesh(filename, V, F);
    V_orig = V;
    V_def=V;
    viewer.data().set_mesh(V, F);

    //precomputations on mesh
    igl::adjacency_list(F, adj_list);
    igl::cotmatrix_entries(V_orig, F, Cov);//for rotation 
    igl::cotmatrix(V_orig, F, L);//write into global L
    spokesAndRimsEdges_sr(V_orig, F, L, edgeSets);
    igl::AABB<Eigen::MatrixXd, 3> tree;
    tree.init(V, F);
    Eigen::Array<double, Eigen::Dynamic, 1> and_visible = Eigen::Array<double, Eigen::Dynamic, 1>::Zero(V.rows());

    factorize(viewer, lambda);    
    
    //initialize handles 
    //id to -1 because nothing was assigned yet
    handle_id.setConstant(V.rows(), 1, -1);
    //handle plugins for deformation, selection, menu
    igl::opengl::glfw::imgui::ImGuiPlugin imgui_plugin;
    viewer.plugins.push_back(&imgui_plugin);
    //add menu
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    imgui_plugin.widgets.push_back(&menu);
    // Add a 3D gizmo plugin
    guizmo.operation = ImGuizmo::TRANSLATE;
    imgui_plugin.widgets.push_back(&guizmo);
    guizmo.visible = false;
    guizmo.T.block(0, 3, 3, 1) = V.row(plugin_vertex).transpose().cast<float>();
    //Add selection plugin
    //selection.mode = igl::opengl::glfw::imgui::SelectionWidget::OFF;
    //imgui_plugin.widgets.push_back(&selection);

    // Attach callback to apply imguizmo's transform to mesh
    guizmo.callback = [&](const Eigen::Matrix4f& T)
    {
        T0.block(0, 3, 3, 1) = pluginpos.transpose().cast<float>();
        const Eigen::Matrix4d TT = (T * T0.inverse()).cast<double>().transpose();
        V_def = (V.rowwise().homogeneous() * TT).rowwise().hnormalized();
        int id = handle_id(plugin_vertex);
        if (handle_vertex_positions.rows() > 0) {
            for (int i = 0; i < handle_vertices.rows(); i++) {
                if (handle_id(handle_vertices(i)) == id) {//this handle
                    handle_vertex_positions.row(i) = V_def.row(handle_vertices(i));//update handle pos
                }
            }
            solve(viewer);
        }
        T0 = T;
        pluginpos = ((pluginpos.rowwise().homogeneous() * TT).rowwise().hnormalized()).eval();
    };

    //draw menu
    menu.callback_draw_viewer_menu = [&]()
    {
        // Add new group
        if (ImGui::CollapsingHeader("Deformation Controls", ImGuiTreeNodeFlags_DefaultOpen))
        {
            int handle_type = static_cast<int>(handle_option);
            if (ImGui::Combo("Handle Option", &handle_type, "VERTEX\0REMOVE\0NONE\0"))
            {
                handle_option = static_cast<Handle>(handle_type);
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
            if (ImGui::Button("10 iterations", ImVec2(-1, 0)))
            {
                viewer.callback_key_pressed(viewer, 'c', 0);
            }
            ImGui::InputText("File name", save_name);
			if (ImGui::Button("save .obj", ImVec2(-1, 0)))
			{
				//save mesh
				std::fstream s{ "../res/" + save_name + ".obj", s.binary | s.trunc | s.in | s.out };
				for (int i = 0; i < V.rows(); i++) {
					s << "v " << V(i, 0) << " " << V(i, 1) << " " << V(i, 2) << std::endl;
				}
				for (int i = 0; i < F.rows(); i++) {
					s << "f " << F(i, 0) + 1 << "/" << F(i, 0) + 1 << " "
						<< F(i, 1) + 1 << "/" << F(i, 1) + 1 << " "
						<< F(i, 2) + 1 << "/" << F(i, 2) + 1 << " " << std::endl;
				}
				s.close();
            }
        }
    };

    // Maya-style keyboard shortcuts for operation
    viewer.callback_key_pressed = [&](decltype(viewer)&, unsigned int key, int mod)
    {
        vertex_picking_mode = false;
        handle_deleting_mode = false;
        switch (key)
        {
        case 'T': case 't': guizmo.operation = ImGuizmo::TRANSLATE;  transform_mode = TRANSLATE;  return true;
        case 'V': case 'v': vertex_picking_mode = false; handle_option = NONE;  return true;
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
                int point_face_idx; //find argmax
                double max = bc.maxCoeff(&point_face_idx);
                
                int point = F(fid, point_face_idx);//indexes into V
                
                if (!handle_deleting_mode) { //now add that vertex to handle
                    //check if already exists
                    if (handle_id(point) != -1) {
                        plugin_vertex = point;
                        guizmo.visible = true;
                        compute_handle_centroid(pluginpos, handle_id(point));
                        guizmo.T.block(0, 3, 3, 1) = pluginpos.transpose().cast<float>();//position
                        return true;
                    }
                    //is new vertex
                    Eigen::VectorXd pt = V.row(point);
                    viewer.data().add_points(V.row(point), point_color);
                    
                    ++num_handles;
                    Eigen::VectorXi up_handle_vertices(num_handles);
                    up_handle_vertices << handle_vertices, point;
                    up_handle_vertices.swap(handle_vertices);
                    
                    Eigen::MatrixXd up_handle_pos (num_handles, 3);
                    up_handle_pos << handle_vertex_positions, V.row(point);
                    up_handle_pos.swap(handle_vertex_positions);
                    
                    
                    assert(constrainedSolver);
                    constrainedSolver->addConstraint(point);
                    
                    
                    plugin_vertex = point;
                    handle_id(plugin_vertex) = plugin_vertex;//set id to itself
                   
                    // factorize(viewer, lambda);//not needed for updating solver!
                    //now make plugin 'active' at that location
                    guizmo.visible = true;
                    pluginpos = V.row(point);
                    guizmo.T.block(0, 3, 3, 1) = pluginpos.transpose().cast<float>();//position
                }
                else {
                    
                    // is 'point' part of the handle set
                    int id = -1;
                    for(int i = 0; i < handle_vertices.size(); ++i) {
                        if(handle_vertices(i) == point) {
                            id = i;;
                            break;
                        }
                    }
                    
                    if(id != -1) {
                        --num_handles;
                        Eigen::VectorXi up_handle_vertices(num_handles);
                        up_handle_vertices.topRows(id) = handle_vertices.topRows(id);
                        up_handle_vertices.bottomRows(num_handles - id) = handle_vertices.bottomRows(num_handles - id);
                        up_handle_vertices.swap(handle_vertices);
                        
                        Eigen::MatrixXd up_handle_pos (num_handles, 3);
                        up_handle_pos.topRows(id) = handle_vertex_positions.topRows(id);
                        up_handle_pos.bottomRows(num_handles - id) = handle_vertex_positions.bottomRows(num_handles - id);
                        up_handle_pos.swap(handle_vertex_positions);
                        
                        constrainedSolver->removeConstraint(point);
                        solve(viewer);
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
T,t   translate
P, p  Click to select a vertex
X, x  Click on handle to remove
v     Switch off handle selection
)";

    //set up viewer
    if(filename.compare("../data/spot.obj")==0||filename.compare("../data/blub.obj")==0){//textured meshes
        mesh_color.setOnes();
        point_color.setZero();
    }
    viewer.data().set_colors(mesh_color);//paint color
    if(filename.compare("../data/spot.obj")==0||filename.compare("../data/blub.obj")==0){
        Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> R,G,B,A;
        if(filename.compare("../data/spot.obj")==0){
            igl::stb::read_image("../data/spot_texture.png",R,G,B,A);
        }
        else{
            igl::stb::read_image("../data/blub_texture.png",R,G,B,A);
        }
        MatrixXd VT, CN, FN;
        MatrixXi FT;
        igl::readOBJ(filename, V, VT, CN, F, FT, FN);
        viewer.data().set_uv(VT, FT);
        viewer.data().show_texture = true;
        viewer.data().set_texture(R,G,B);
    }
    viewer.data().set_vertices(V);//todo remove
    viewer.data().compute_normals();
    viewer.data().show_lines = false;
    viewer.data().point_size = 15;
    viewer.core().background_color.setOnes();
    viewer.launch();
}
