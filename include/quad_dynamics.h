#ifndef QUAD_DYNAMICS
#define QUAD_DYNAMICS

//#include "UDPConnect.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include<Eigen/SparseCholesky>
#include <unsupported/Eigen/MatrixFunctions>
#include <unsupported/Eigen/KroneckerProduct>
#include<cmath>
#include<chrono>


class Quadcopter{
public:
    Quadcopter();
    ~Quadcopter();
    const int nx = 12;
    const int nu = 4;
    const int ny = 10;
    const int nt = 6;
    
    Eigen::MatrixXd A; // State matrix
    Eigen::MatrixXd B; // Input matrix
    Eigen::MatrixXd C; // State output matrix
    Eigen::MatrixXd D; // Input output matrix
    Eigen::MatrixXd K; // Feedback matrix
    Eigen::MatrixXd x0; // Initial state
    ; // Initial input
    
    void set_x0(double * x0);
    void step(Eigen::VectorXd & u0);
    
    void load_system(Eigen::MatrixXd &A0, Eigen::MatrixXd &Ai,
		     Eigen::MatrixXd &Aij, Eigen::MatrixXd &AN);

    void load_costs(Eigen::MatrixXd &Q0, Eigen::MatrixXd &Qi, Eigen::MatrixXd &QN);

};


#endif // define QUAD_DYNAMICS
