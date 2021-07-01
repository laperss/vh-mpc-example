cmake_minimum_required(VERSION 3.1)
include(ExternalProject)
add_compile_options(-std=c++14 -Ofast)

project(vhmpc-example)

find_package(osqp REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

include_directories(${Boost_INCLUDE_DIR} ${Boost2_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})

# TEST INCREASING N ========================================================================= 
SET (SRCS
  ./VH_MPC_test_nominal.cpp
  ./src/quad_dynamics.cpp
  ./src/VH_MPC.cpp)
add_executable(vh_mpc_test_nominal ${SRCS} )
target_link_libraries (vh_mpc_test_nominal Eigen3::Eigen osqp::osqp   ${Boost_LIBRARIES} )
target_include_directories(vh_mpc_test_nominal PUBLIC include /home/laperss/Programs/osqp/lin_sys/direct/qdldl  ./dji_export/)

SET (SRCS
  ./VH_MPC_test_recursive.cpp
  ./src/quad_dynamics.cpp
  ./src/VH_MPC.cpp)
add_executable(vh_mpc_test_recursive ${SRCS} )
target_link_libraries (vh_mpc_test_recursive Eigen3::Eigen osqp::osqp   ${Boost_LIBRARIES} )
target_include_directories(vh_mpc_test_recursive PUBLIC include /home/laperss/Programs/osqp/lin_sys/direct/qdldl  ./dji_export/)
