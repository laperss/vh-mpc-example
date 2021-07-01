#include "VH_MPC.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <clocale>
#include <thread>

#include "quad_dynamics.h" // contains forward declarations and nx, ny....


Quadcopter::Quadcopter(){
    x0.resize(nx,1);
    
    A.resize(nx,nx);
    B.resize(nx,nu);
    K.resize(nu,nx);
    C.resize(ny,nx);
    D.resize(ny,nu);
}


Quadcopter::~Quadcopter(){
}

void get_lqr_feedback(const Eigen::Matrix<double, 12, 12> A,
                      const Eigen::Matrix<double, 12, 4>  B,
                      const Eigen::Matrix<double, 12, 12> Q,
                      const Eigen::Matrix<double, 4, 4>   R,
                      Eigen::MatrixXd & Qf,
                      Eigen::MatrixXd & K){
    Qf = Q;
    Eigen::Matrix<double, 12, 12> Pnew;

    for (int i=0;i<100;i++){
	Eigen::LLT<Eigen::MatrixXd> llt(R+B.transpose()*Qf*B);
	
        Pnew = Q + A.transpose()*Qf*A  -A.transpose()*Qf*B*llt.solve(B.transpose()*Qf*A);
        Qf = Pnew;	
    }

    K = -(R + B.transpose()*Qf*B).inverse()*B.transpose()*Qf*A;
    
}

void Quadcopter::set_x0(double * x0){    
    for (int i=0;i<nx;i++)
	this->x0(i,0) = x0[i];
}


void Quadcopter::step(Eigen::VectorXd & u0){
    x0 = A*x0 + B*u0;
}

void Quadcopter::load_system(Eigen::MatrixXd &A0, Eigen::MatrixXd &Ai,
			     Eigen::MatrixXd &Aij, Eigen::MatrixXd &AN){ 
    
    this->B = Eigen::MatrixXd::Zero(nx,nu);
    this->C = Eigen::MatrixXd::Zero(ny,nx);
    this->D = Eigen::MatrixXd::Zero(ny,nu);
    
    Eigen::MatrixXd At = Eigen::MatrixXd::Zero(nt,nx);

    this->A <<
        1,   0,   0,   0.0995017,           0,           0,            0,   0.0384679,   0,            0,  0.00097833,           0,
        0,   1,   0,           0,   0.0995017,           0,   -0.0384877,           0,   0, -0.000972394,           0,           0,
        0,   0,   1,           0,           0,   0.0884798,            0,           0,   0,            0,           0,           0,
        0,   0,   0,     0.99005,           0,           0,            0,    0.729704,   0,            0,   0.0261409,           0,
        0,   0,   0,           0,     0.99005,           0,    -0.730185,           0,   0,   -0.0258854,           0,           0,
        0,   0,   0,           0,           0,      0.7788,            0,           0,   0,            0,           0,           0,
        0,   0,   0,           0,           0,           0,     0.741132,           0,   0,    0.0488681,           0,           0,
        0,   0,   0,           0,           0,           0,            0,     0.73858,   0,            0,   0.0495624,           0,
        0,   0,   0,           0,           0,           0,            0,           0,   1,            0,           0,   0.0633333,
        0,   0,   0,           0,           0,           0,     -3.95832,           0,   0,     0.107801,           0,           0,
        0,   0,   0,           0,           0,           0,            0,    -4.01455,   0,            0,    0.114095,           0,
        0,   0,   0,           0,           0,           0,            0,           0,   0,            0,           0,    0.366667;

    this->B <<
        0,         0,           0.00208787, 0,
        0,        -0.00192582,  0,          0,
        0.0115202, 0,           0,          0,
        0,         0,           0.0808297,  0,
        0,        -0.0748257,   0,          0,
        0.2212,    0,           0,          0,
        0,         0.245925,    0,          0,
        0,         0,           0.266648,   0,
        0,         0,           0,          0.0366667,
        0,         3.7604,      0,          0,
        0,         0,           4.09484,    0,
        0,         0,           0,          0.633333;


    

    
    this->C.block(0,3,3,3)  = Eigen::MatrixXd::Identity(3,3);
    this->C.block(3,6,3,3)  = Eigen::MatrixXd::Identity(3,3);
    this->D.block(6,0,4,4)  = Eigen::MatrixXd::Identity(4,4);


    
    std::cout << "A\n" << this->A << std::endl;
    std::cout << "B\n" << this->B << std::endl;
    std::cout << "C\n" << this->C << std::endl;
    std::cout << "D\n" << this->D << std::endl;
    
    At.block(0,0,3,3) = Eigen::MatrixXd::Identity(3,3);
    At.block(3,3,3,3) = Eigen::MatrixXd::Identity(3,3);

    A0 = Eigen::MatrixXd::Zero(ny+nx,nu);
    Ai = Eigen::MatrixXd::Zero(ny+nx,nx+nu);
    Aij = Eigen::MatrixXd::Zero(ny+nx,nx+nu);
    AN = At;

    A0.block(0,0,ny,nu) = this->D;
    A0.block(ny,0,nx,nu) = this->B;
    Ai.block(0,0,ny,nx) = this->C;
    Ai.block(0,nx,ny,nu) = this->D;
    Ai.block(ny,0,nx,nx) = this->A;
    Ai.block(ny,nx,nx,nu) = this->B;
    Aij.block(ny,0,nx,nx) = -Eigen::MatrixXd::Identity(nx,nx);
    
}
void Quadcopter::load_costs(Eigen::MatrixXd &Q0, Eigen::MatrixXd &Qi, Eigen::MatrixXd &QN){

    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(nx,nx)*0.1;


    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(nu,nu)*5;
    //Eigen::MatrixXd Qf = Eigen::MatrixXd::Identity(nx,nx)*1;
    Eigen::MatrixXd Qf(nx,nx);
    get_lqr_feedback(this->A, this->B, Q, R, Qf, K);    

    Q0.block(0,0,nu,nu) = R;
    Qi = Eigen::MatrixXd::Zero(nx+nu,nx+nu);
    QN = Eigen::MatrixXd::Zero(nx,nx);
    for (int i=0;i<nx;i++)
	for (int j=i;j<nx;j++)
	    QN(i,j) = Qf(i,j);
    Qi.block(0,0,nx,nx) = Q;
    Qi.block(nx,nx,nu,nu) = R;

    std::cout << "Qf\n" << Qf << std::endl;
    
    return;
}

