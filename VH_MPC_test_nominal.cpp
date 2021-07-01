#include "VH_MPC.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <clocale>
#include <thread>
#include <iostream>
#include <fstream>

#include "quad_dynamics.h"


int main() {
    std::string filename = "results25_nominal";
    std::string fileextension = ".txt";
    
    Quadcopter quad;
    int nx = quad.nx;
    int nu = quad.nu;
    int ny = quad.ny;
    int nt = quad.nt;
    
    double initial_state[nx] = {-7.5, 6.5, 7.0, 0.1,0.2,0.3,0,0,0.0,0,0,0};

    // System matrices (A, B) and mode 2 feedback (K)
    Eigen::MatrixXd Amat(nx,nx);
    Eigen::MatrixXd Bmat(nx,nu);
    Eigen::MatrixXd Kmat(nu,nx);

    // The matrices that will be a part of the KKT matrix (see paper)
    Eigen::MatrixXd A0_f(ny+nx,nu);
    Eigen::MatrixXd Ai_f(ny+nx,nx+nu);
    Eigen::MatrixXd Aij_f(ny+nx,nx+nu);
    Eigen::MatrixXd AN_f(nt,nx);
    
    Eigen::MatrixXd Q0_f(nu,nu);
    Eigen::MatrixXd Qi_f(nx+nu,nx+nu);
    Eigen::MatrixXd QN_f(nx,nx);

    // Setup the quadcopter
    quad.set_x0(initial_state);
    quad.load_system(A0_f, Ai_f, Aij_f, AN_f);
    quad.load_costs(Q0_f, Qi_f, QN_f);
    Amat = quad.A;
    Bmat = quad.B;
    Kmat = quad.K;

    // Setup the MPC
    nominal_MPC nominal_mpc(nx,nu,ny,nt);    
    nominal_mpc.set_Q0(Q0_f);
    nominal_mpc.set_Qi(Qi_f);
    nominal_mpc.set_QN(QN_f);
    nominal_mpc.set_A0(A0_f);
    nominal_mpc.set_Ai(Ai_f);
    nominal_mpc.set_Aij(Aij_f);
    nominal_mpc.set_AN(AN_f);
    
    nominal_mpc.set_Amat(Amat);
    nominal_mpc.set_Bmat(Bmat);
    nominal_mpc.set_Kmat(Kmat);    
    
    // Load the MPC settings
    OSQPSettings * settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));
    load_settings(settings);
    nominal_mpc.settings = settings;
    
    double v_max = 3.0 ; // m/s
    double attitude_lim = 0.2;
    double input_max = 0.2;
    Eigen::MatrixXd by_l(ny,1);
    Eigen::MatrixXd by_u(ny,1);
    Eigen::MatrixXd bt_l(nt,1);
    Eigen::MatrixXd bt_u(nt,1);

    by_u << v_max, v_max, v_max, attitude_lim, attitude_lim, attitude_lim,
	v_max, input_max, input_max, input_max;
    by_l = -by_u;
    bt_l << -0.5, -0.5, -0.1, -0.5, -0.5, -0.5 ;
    bt_u << 0.5, 0.5, 0.1, 0.5, 0.5, 0.5; //, 0.5, 0.5, 0.5;

    nominal_mpc.set_z_bound(by_l,by_u);
    nominal_mpc.set_terminal_bound(bt_l,bt_u);    
    
    // Set the initial state
    Eigen::MatrixXd x0(nx,1);
    for (int i=0;i<nx;i++){
	x0(i) = initial_state[i];
    }


    double setup_time_nominal;
    VHMPC_Result results;
    
    
    int Nmax = 250;
    nominal_mpc.initialize(Nmax);
    
    double primal_solution[(nx+nu)*Nmax]={0};
    double dual_solution[(nx+ny)*Nmax+nt]={0};



    Eigen::MatrixXd Ax0(nx,1);

    Eigen::MatrixXd state(nx,1);
    Eigen::MatrixXd state_next(nx,1);
    Eigen::MatrixXd input(nu,1);

    int N_initial = 40;
    int N_final = 90;
    int N0 = N_initial;
    int num_tests =20;
    int return_val;
    for(int test_no=0;test_no<num_tests;test_no++){
	nominal_mpc.load_constraints();	
	return_val = nominal_mpc.setup(N_initial, Nmax);
	setup_time_nominal = nominal_mpc.setup_time;

	
	// These are setup before solving	
	printf("Nominal %i\n", test_no);
	printf("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
	printf("  status  iter   nnz    t setup   t solve   t total    t ADMM   obj_val\n");
	        
	int iter = 0;
	for (int N1 = N_initial; N1 < N_final ; N1++){
	    state = x0;

	    if (iter>0){
		nominal_mpc.compute_warm_start(N0, N1,primal_solution, dual_solution);
		nominal_mpc.cleanup_workspace();
	    }
	    else settings->warm_start = false;
	    
	    Ax0 = -Amat*x0;

	    // Solve the nominal optimization problem -----------------------------------
	    return_val = nominal_mpc.solve_N(Ax0, N1, 
                                             primal_solution, dual_solution, 
                                             results);
	    printf("%3i  %i   %5i   %4i    %7.6f  %7.6f  %7.6f  %7.6f  %6.3f\n", N1,
		   results.status_val, results.niter, results.Lnnz,
		   results.update_time,results.solve_time,
		   results.update_time+results.solve_time,
		   results.osqp_solve_time, results.objective_value);
	    std::this_thread::sleep_for(std::chrono::milliseconds(100));
	

	
	    if (results.status_val != 1){
		printf("Problem not solvable: %i\n", results.status_val);
	    }
	    N0 = N1;
	    iter++;
	} // End of iter N


    

	nominal_mpc.cleanup_workspace();
	nominal_mpc.cleanup_data();
	
    } // End of test
    
    free(settings);

}
