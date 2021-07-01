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
    // LOAD THE DYNAMICAL SYSTEM, COST, AND CONSTRAINTS
    
    Quadcopter quad;
    int nx = quad.nx;
    int nu = quad.nu;
    int ny = quad.ny;
    int nt = quad.nt;
    
    double initial_state[nx] = {-7.5, 6.5, 7.0, 0.1,0.2,0.3,0,0,0.0,0,0,0};
    double initial_input[nu] = {0,0,0,0};
    quad.set_x0(initial_state);

    Eigen::MatrixXd A0_f(ny+nx,nu);
    Eigen::MatrixXd Ai_f(ny+nx,nx+nu);
    Eigen::MatrixXd Aij_f(ny+nx,nx+nu);
    Eigen::MatrixXd AN_f(nt,nx);

    Eigen::MatrixXd Q0_f(nu,nu);
    Eigen::MatrixXd Qi_f(nx+nu,nx+nu);
    Eigen::MatrixXd QN_f(nx,nx);

    Eigen::MatrixXd Amat(nx,nx);
    Eigen::MatrixXd Bmat(nx,nu);
    Eigen::MatrixXd Kmat(nu,nx);
    
    quad.load_system(A0_f, Ai_f, Aij_f, AN_f);
    quad.load_costs(Q0_f, Qi_f, QN_f);
    Amat = quad.A;
    Bmat = quad.B;
    Kmat = quad.K;

    recursive_MPC recursive_mpc(nx,nu,ny,nt);    
    recursive_mpc.set_Q0(Q0_f);
    recursive_mpc.set_Qi(Qi_f);
    recursive_mpc.set_QN(QN_f);
    recursive_mpc.set_A0(A0_f);
    recursive_mpc.set_Ai(Ai_f);
    recursive_mpc.set_Aij(Aij_f);
    recursive_mpc.set_AN(AN_f);    

    recursive_mpc.set_Amat(Amat);
    recursive_mpc.set_Bmat(Bmat);
    recursive_mpc.set_Kmat(Kmat);

    // Load the MPC settings
    OSQPSettings * settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));
    load_settings(settings);
    recursive_mpc.settings = settings;
    int return_val;
    double v_max = 3.0 ; // m/s
    double attitude_lim = 0.2;
    double input_max = 0.2;
    Eigen::MatrixXd by_l(ny,1);
    Eigen::MatrixXd by_u(ny,1);
    Eigen::MatrixXd bt_l(nt,1);
    Eigen::MatrixXd bt_u(nt,1);

    by_u << v_max, v_max, v_max,
	attitude_lim, attitude_lim, attitude_lim,
	v_max, input_max, input_max, input_max;
    by_l = -by_u;
    bt_l << -0.5, -0.5, -0.1, -0.5, -0.5, -0.5 ;
    bt_u << 0.5, 0.5, 0.1, 0.5, 0.5, 0.5; //, 0.5, 0.5, 0.5;

    recursive_mpc.set_z_bound(by_l,by_u);
    recursive_mpc.set_terminal_bound(bt_l,bt_u);
    

    double setup_time_recursive;

    settings->rho = 5.0;

    // These are setup before solving

    VHMPC_Result results;
    
    int Nmax = 250;
    recursive_mpc.initialize(Nmax);
    
    double primal_solution[(nx+nu)*Nmax]={0};
    double dual_solution[(nx+ny)*Nmax+nt]={0};


    Eigen::MatrixXd Ax0(nx,1);
    Eigen::MatrixXd state(nx,1);
    Eigen::MatrixXd state_next(nx,1);
    Eigen::VectorXd input(nu);



    int N_initial = 100;
    int N0 = N_initial;    
    int num_tests = 20;
    for(int test_no=0;test_no<num_tests;test_no++){
	// Reset quadcopter
	quad.set_x0(initial_state);
        for (int i=0;i<nu;i++)
            input(i) = initial_input[i];
	
	// These are setup before solving
	recursive_mpc.load_constraints();		
	return_val = recursive_mpc.setup(N_initial, Nmax);
	setup_time_recursive = recursive_mpc.setup_time;
	printf("Setup recursive (%i):  %f\n", return_val, setup_time_recursive);

	printf("Recursive\n");
	printf("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
	printf("     u0    u1     u2    x0    x1    x2  status iter    nnz    t setup   t solve   t total    t ADMM  obj_val |status iter   nnz    t setup   t solve   t total    t ADMM obj_val\n");
    


        
	// Try updating all three 

	int iter = 0;
	for (int N1 = 0; N1 < 100; N1++){
	    state = quad.x0;
	
	    if (iter!=0){
		recursive_mpc.compute_warm_start_shifted(recursive_mpc.Nprev, recursive_mpc.Nmin,
						       primal_solution, dual_solution);
	    }
	    else {
		recursive_mpc.disable_warm_start();
		settings->warm_start = false;

	    }

	    Ax0 = -Amat*state;
	
	    // Solve the recursive LDL optimization problem	 ----------------------------	
	    return_val = recursive_mpc.solve_variable_horizon(Ax0, N1, 
							      primal_solution, dual_solution, 
							      results);
	
	    std::this_thread::sleep_for(std::chrono::milliseconds(100));


	    printf("%3i  ", iter);
	    printf("%3.2f  %3.2f  %3.2f  ", input(0), input(1), input(2));
	    printf("%2.2f  %2.2f  %2.2f  ", state(0), state(1), state(2));

	    printf("%2i  %4i  %4i  %7.6f  %7.6f  %7.6f  %7.6f  %6.3f\n",
		   results.status_val, results.niter,
		   results.Lnnz,
		   results.update_time,results.solve_time,
		   results.update_time+results.solve_time,
		   results.osqp_solve_time, results.objective_value);


	
	    if (results.status_val !=1){
		printf("Problem not solvable: %i\n",results.status_val);
	    }
	    
	    // STEP IN SIMULATION
	    for (int j=0;j<nu;j++) input(j) = recursive_mpc.optimal_trajectory[j][0];
	    quad.step(input);
/*
	    printf("new  ");
	    printf("%2.2f  %2.2f  %2.2f  ", quad.u0(0), quad.u0(1), quad.u0(2));
	    printf("%2.2f  %2.2f  %2.2f  \n", quad.x0(0), quad.x0(1), quad.x0(2));
*/	    
	    iter++;
	} // End of iter N


	recursive_mpc.cleanup_workspace();
	recursive_mpc.cleanup_data();

    } // End of test
    
    free(settings);

}
