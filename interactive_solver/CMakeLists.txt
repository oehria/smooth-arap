cmake_minimum_required(VERSION 3.16)
project(smooth_arap)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Libigl
include(libigl)

# Enable the target igl::glfw
igl_include(glfw)
# Other modules you could enable
#igl_include(embree)
igl_include(imgui)
#igl_include(opengl)
igl_include(stb)
#igl_include(predicates)
#igl_include(xml)
#igl_include(copyleft cgal)
#igl_include(copyleft comiso)
#igl_include(copyleft core)
#igl_include(copyleft cork)
#igl_include(copyleft tetgen)
#igl_include(restricted matlab)
#igl_include(restricted mosek)
#igl_include(restricted triangle)

# Add your project files
file(GLOB SRC_FILES *.cpp)
add_executable(${PROJECT_NAME} ${SRC_FILES})

# Link igl (and the glfw module) to your project
target_link_libraries(${PROJECT_NAME} PUBLIC 
  igl::imgui
  igl::glfw
  ## Other modules you could link to
  # igl::embree
  # igl::opengl
  igl::stb
  # igl::predicates
  # igl::xml
  # igl_copyleft::cgal
  # igl_copyleft::comiso
  # igl_copyleft::core
  # igl_copyleft::cork
  # igl_copyleft::tetgen
  # igl_restricted::matlab
  # igl_restricted::mosek
  # igl_restricted::triangle
  )
  
find_path(SUITESPARSE_INCLUDE_DIR cholmod.h PATHS /usr/include/suitesparse)
find_library(SUITESPARSE_LIB cholmod PATHS /usr/lib /usr/lib/x86_64-linux-gnu)
find_library(SUITESPARSECONFIG_LIB suitesparseconfig PATHS /usr/lib /usr/lib/x86_64-linux-gnu)

if (SUITESPARSE_INCLUDE_DIR AND SUITESPARSE_LIB AND SUITESPARSECONFIG_LIB)
    target_include_directories(smooth_arap PRIVATE ${SUITESPARSE_INCLUDE_DIR})
    target_link_libraries(smooth_arap PRIVATE ${SUITESPARSE_LIB} ${SUITESPARSECONFIG_LIB})
else()
    message(FATAL_ERROR "SuiteSparse libraries not found")
endif()

