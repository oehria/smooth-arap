# Smooth ARAP

This is the implementation of the paper "Higher Order Continuity for Smooth As-Rigid-As-Possible Shape modeling" by Annika Oehri, Philipp Herholz and Olga Sorkine-Hornung. 

**BibTex**
```
to be released
```

## Installation guide

The code is written in C++ and its functionality only depends on [libigl](https://github.com/libigl/libigl) (v2.5.0), for more explanations on the compilation of libigl projects please refer to [this](https://libigl.github.io/). With an existing libigl folder, you should be able to compile and run the project via the standard CMake-routine:
```
mkdir build
cd build
cmake ..
make
```
For increased efficiency, we have also conducted experiments swapping out the Eigen solver for the CHOLMOD one, to make use of this option you should further install the [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse) library. Otherwise, you can disable this by deleting the inclusion of the library in the corresponding CMakeLists.txt and disable the USE_CHOLMOD macro. Make sure to also change to the desired solver (by default the Eigen one is used for simplicity).

## Interactive
There are two versions of this published, one being the `normal` interactive application, where users can select handles and drag them, causing the mesh to deform accordingly. `Interactive_solver` contains the code for the solver updates, that avoids having to refactorize when adding a new handle. For simplicity, we implemented this in a more limited framework, most notably only point handles are available (but conceptually it can be extended to the full functionality).

**Input:** A manifold, orientable triangle mesh to deform through its filename, e.g. `my_mesh.obj`, which is assumed to be in the data folder. 

**Interface:**
* Transformation mode: Change between translating, rotating and scaling handles. For translation you can also press the shortcut key `t` and for rotation `r`. 
* Handle option: Choose between creating a handle with the lasso tool (`l`), marque area (`m`), single vertex handle (`p`), removing a vertex (`x`) or creating no new handle (`v`) via the dropdown menu or shortcut keys
* Enter the desired smoothness parameter lambda
* Move the handles around as you desire and the mesh will deform accordingly. Should you want to let it converge further in between, use the `10 iterations` button or press the shortcut `c`
* File name: in case you want to save your results in the res folder (make sure it exists and is next to the data folder), specify the name under which you want to save them
* Press `save .obj`: This saves your deformed result  as a .obj file

The interface for the interactive application with the more efficient solver is the same, there are just fewer options available.

## Non-Interactive
This code serves the purpose of recreating the deformation survey [Sorkine and Botsch 2009] examples with our method. 

**Input:** The mesh to deform, should be one of the survey examples, so `knubbel.off`, `cylinder.off`, `bar.off` or `cactus.off`. The constraints are already in the code and will be automatically selected based on the chosen mesh.

**Interface:**
* Smoothness Lambda: enter your desired smoothness value
* Initialization Scheme: Choose between Handle (initializing via the original mesh), Poisson (solving a constrained Poisson system) and Bi-Laplacian (solving a constrained Bi-Laplacian system). The result will be displayed. 
* Max: Adjust the maximal number of iterations if needed
* Press `arap deformation` button to run the deformation until convergence (relative change of mesh < 0.0001) or the maximum number of iterations is reached
* File name: in case you want to save your results in the res folder (make sure it exists and is next to the data folder), specify the name under which you want to save them
* Press `save .obj and stats`: This saves your mesh as a .obj and an additional .txt file with your runtime and number of iterations until convergence. 
