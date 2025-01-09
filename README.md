# Smooth ARAP

This is the implementation of the paper "Higher Order Continuity for Smooth As-Rigid-As-Possible Shape modeling" by Annika Oehri, Philipp Herholz and Olga Sorkine-Hornung published in TODO. 

**BibTex**
```
TODO
```

## Installation guide

The code is written in C++ and only depends on [libigl](https://github.com/libigl/libigl) (v2.5.0), for more explanations on the compilation of libigl projects please refer to [this](https://libigl.github.io/). With an existing libigl folder, you should be able to compile and run the project via the standard CMake-routine:
```
mkdir build
cd build
cmake ..
make
```

## Interactive
There are two versions of this published, one being the normal interactive application, where users can select handles and drag them, causing the mesh to deform accordingly. Interactive_solver contains the code for the solver updates, that avoids having to refactorize when adding a new handle. As this is only more efficient for small changes, this framework is limited to single point handles. 
Otherwise, both work the same way.

**Input:** A manifold, orientable triangle mesh to deform. 

**How to use:**
* Start the application with the path to the mesh as the first command line argument
* Select handles and change their positions via translation, rotation and scaling.
* Press `3` to initialize the parameterization via LSCM
* Press `6`to compute the Chebyshev parameterization. This will automatically run until convergence/ the iteration limit is reached. To adapt these, adapt convergence_precision and max_iters in the code. If you want the shearing to be limited, set the shearing angle limit in the GUI beforehand.
* Press `4`to run ARAP parameterization. Again, this automatically runs until convergence.
* You can adapt the Texture Resolution by changing the number in the GUI
* To run from a saved initialization, you can use the button `load texture from .obj` and then run Chebyshev parameterization as per usual.
* To save your results with corresponding metrics like Chebyshev error, runtime etc, enter a name and click on `save .obj and stats`. It will write into the `res` folder.


## Non-Interactive
This code serves the purpose of recreating the deformation survey [Sorkine and Botsch 2009] examples with our method. 

**Input:** The mesh to deform, should be one of the survey examples, so `knubbel.off`, `cylinder.off`, `bar.off` or `cactus.off`. The constraints are already in the code and will be automatically selected based on the chosen mesh.

**Interface:**
* Smoothness Lambda: enter your desired smoothness value
* Initialization Scheme: Choose between Handle (initializing via the original mesh), Poisson (solving a constrained Poisson system) and Bi-Laplacian (solving a constrained Bi-Laplacian system). The result will be displayed. 
* Max: Adjust the maximal number of iterations if needed
* Press `arap deformation` button to run the deformation until convergence (relative change of mesh < 0.0001) or the maximum number of iterations is reached
* File name: in case you want to save your results, specify the name under which you want to save them
* Press `save .obj and stats`: This saves your mesh as a .obj and an additional .txt file with your runtime and number of iterations until convergence. 
