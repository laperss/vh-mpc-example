#ifndef VH_MPC
#define VH_MPC

#include "osqp.h"
#include "recursive_ldl.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include<Eigen/SparseCholesky>
#include <unsupported/Eigen/MatrixFunctions>
#include <unsupported/Eigen/KroneckerProduct>
#include<cmath>
#include<chrono>


typedef struct {
  c_int   niter;
  c_int   status_val;
  c_int   status_polish;
  c_float objective_value;
  c_float pri_res;
  c_float dua_res;
  c_float solve_time;
  c_float update_time;
  c_float osqp_setup_time;
  c_float osqp_solve_time;
  c_float osqp_update_time;
  c_float osqp_polish_time;
  c_float osqp_run_time;
  c_float osqp_admm_1;
  c_float osqp_admm_2;
  c_float osqp_admm_3;
  c_int   Ln;
  c_int   Lm;
  c_int   Lnnz;
  c_float   rho_estimate;
  c_int   rho_updates;
  c_int horizon;
  c_int num_opt;    
} VHMPC_Result;





void array_to_csc(csc* C, const double * matrix);

void load_settings(OSQPSettings * settings);


class VHMPC{
protected:
    c_float epsilon = 1e-7;
    int last_status;
    int Nmax;

    int nx;
    int ny;
    int nu;
    int nt;
    csc * Q0;
    csc * Qi;
    csc * QN;
    csc * A0;
    csc * Ai;
    csc * Aij;
    csc * AN;

    Eigen::MatrixXd A;
    Eigen::MatrixXd B;
    Eigen::MatrixXd K;
    
    double * linear_cost;

    // Controlled variable bounds
    double * z_lower;
    double * z_upper;
    
    // Terminal constraint bounds
    double * t_lower;
    double * t_upper;
    
    double * lb; // Lower bound, entire OSQP vector
    double * ub; // Upper bound, entire OSQP vector

    // Optimal solution
    double * primal;
    double * dual;    

    OSQPDataRLDL * data;


    void compute_P_matrix(OSQPData * data, c_int N);
    void compute_A_matrix(OSQPData * data, c_int N);
    c_int update_AP_matrices(OSQPData * data, c_int Nold, c_int Nnew);
    
    void save_solution(int N,
		       double update_time,
		       double solve_time,
		       VHMPC_Result &results);
    
    double compute_cost(double obj_val, int N);
public:
    VHMPC(int nx, int nu, int ny, int nt);
    ~VHMPC();
    void initialize(int Nmax);
    
    int N0;     // Last tested N: for warm starting
    int Nprev;  // Last tested N: for warm starting
    int Nmin;   // Last optimal N
    int deltaN; // Difference between previous optimal and current N
    
    OSQPWorkspace * work;
    OSQPSettings * settings;

    double setup_time;
    double update_time;
    double solve_time;

    double ** optimal_trajectory;

    // Internal functions dependent on method
    virtual int setup(int N, int Nmax) {};
    virtual int update_osqp(int N1) {};
    virtual int solve_osqp(int N1,
			   double * warm_start_vector,
			   double * warm_start_vector_dual, 
			   VHMPC_Result &results) {};

    // General solve methods
    int solve_N(const Eigen::MatrixXd &Ax0, int N1,
	      double * warm_start_vector,
	      double * warm_start_vector_dual, 
	      VHMPC_Result &results);
    
    int solve_variable_horizon(const Eigen::MatrixXd &Ax0, int N1,
			       double * warm_start_vector,
			       double * warm_start_vector_dual, 
			       VHMPC_Result &results);
    
    void cleanup_workspace();
    void cleanup_data();

    // Load the OSQP constraint vectors
    void load_constraints();
    // Update the OSQP constraint vectors from N0 to N1
    void update_initial_constraint(const Eigen::MatrixXd &x0);
    void update_constraints(const int N0, const int N1);
    void update_constraints(const int N0, const int N1,
			    const Eigen::MatrixXd &x0);



    
    void copy_matrix(const Eigen::MatrixXd &M, csc *& N);
    void set_Amat(const Eigen::MatrixXd& A) {this->A = A;}
    void set_Bmat(const Eigen::MatrixXd& B) {this->B = B;}
    void set_Kmat(const Eigen::MatrixXd& K) {this->K = K;}
    void set_Q0(const Eigen::MatrixXd& Q0) {copy_matrix(Q0, this->Q0);}
    void set_Qi(const Eigen::MatrixXd& Qi) {copy_matrix(Qi, this->Qi);}
    void set_QN(const Eigen::MatrixXd& QN) {copy_matrix(QN, this->QN);}
    void set_A0(const Eigen::MatrixXd& A0) {copy_matrix(A0, this->A0);}
    void set_Ai(const Eigen::MatrixXd& Ai) {copy_matrix(Ai, this->Ai);}
    void set_Aij(const Eigen::MatrixXd& Aij) {copy_matrix(Aij, this->Aij);}
    void set_AN(const Eigen::MatrixXd& AN)  {copy_matrix(AN, this->AN);}

    // Set the lower and upper bound of controlled variable z
    void set_z_bound(const Eigen::MatrixXd &lower,const Eigen::MatrixXd &upper);
    void set_terminal_bound(const Eigen::MatrixXd &lower,const Eigen::MatrixXd &upper);

    // Warm start primal and dual from last optimal solution
    void compute_warm_start(int N0,  int N1, double *primal, double *dual);
    void compute_warm_start_shifted(int N0,  int N1, double *primal, double *dual);
    // If primal/dual has already been computed up until N=N0    
    void update_warm_start(int N0,  int N1, double *primal, double *dual);

    // Enable or disable warm start
    void disable_warm_start(){osqp_update_warm_start(work, 0);};
    void enable_warm_start(){osqp_update_warm_start(work, 1);};


};


class nominal_MPC : public VHMPC{
public:
    nominal_MPC(int nx, int nu, int ny, int nt);
    ~nominal_MPC(){};
    //void solve();
    //int setup(){};
    int setup(int N, int Nmax);

    int solve_osqp(int N1,
		   double * warm_start_vector,
		   double * warm_start_vector_dual, 
		   VHMPC_Result &results);


    int update_osqp(int N1);
    
};

class recursive_MPC : public VHMPC{
public:
    recursive_MPC(int nx, int nu, int ny, int nt);
    ~recursive_MPC(){};
    //void solve();
    //int setup(){};
    int setup(int N, int Nmax);

    int solve_osqp(int N1,
		   double * warm_start_vector,
		   double * warm_start_vector_dual, 
		   VHMPC_Result &results);

    int update_osqp(int N1);
    
};

class combined_MPC : public VHMPC{
public:
    combined_MPC(int nx, int nu, int ny, int nt);
    ~combined_MPC(){};
    //void solve();
    //int setup(){};
    int setup(int N, int Nmax);

    int solve_osqp(int N1,
	      double * warm_start_vector,
	      double * warm_start_vector_dual, 
	      VHMPC_Result &results);
    int update_osqp(int N1);
    
};






#endif // define VH_MPC
