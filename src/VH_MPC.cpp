#include "VH_MPC.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <clocale>






void array_to_csc(csc* C, const double * matrix) {
  c_int nz = 0;
  c_float value;
  c_float epsilon = 1e-7;
  c_int rows = C->m;
  c_int count = 0;

  // For every column
  for(int i=0; i <C->n; i++){
    C->p[i] = count;
    for(int j=0;j<rows;j++){
      value = matrix[i*rows + j];
      if (fabs(value)>epsilon){
        C->i[count] = j;
        C->x[count++] = value;
      }
    }
  }
  C->p[C->n] = count;
  return;
}

void load_settings(OSQPSettings * settings){
    osqp_set_default_settings(settings);    
    settings->max_iter = 250; 
    settings->eps_abs = 1e-3; 
    settings->eps_rel = 1e-3;
    settings->eps_prim_inf = 1e-3;
    settings->eps_dual_inf = 1e-3;
    settings->scaling = false;
    settings->verbose = false;
    settings->polish = false;
    settings->warm_start = true;
    settings->adaptive_rho = false;
    settings->linsys_solver = QDLDL_SOLVER;
    settings->rho = 5.0;
    settings->check_termination = 25;
}


VHMPC::VHMPC(int nx, int nu, int ny, int nt): Nmin(-1), Nprev(-1), deltaN(0){
    this->nx = nx;
    this->nu = nu;
    this->ny = ny;
    this->nt = nt;
    
    z_lower = new double[ny];
    z_upper = new double[ny];
    t_lower = new double[nt];
    t_upper = new double[nt];
    



    work = OSQP_NULL;
    data = OSQP_NULL;
    
}

VHMPC::~VHMPC(){
    if (this->Q0) csc_spfree(this->Q0);
    if (this->Qi) csc_spfree(this->Qi);
    if (this->QN) csc_spfree(this->QN);
    if (this->A0) csc_spfree(this->A0);
    if (this->Ai) csc_spfree(this->Ai);
    if (this->Aij) csc_spfree(this->Aij);
    if (this->AN) csc_spfree(this->AN);

    delete[] z_lower;
    delete[] z_upper;
    delete[] t_lower;
    delete[] t_upper;

    delete[] primal;
    delete[] dual;
    
    delete[] lb;
    delete[] ub;

    delete[] linear_cost;

    for (int i=0; i<nx+nu; i++)	delete[] optimal_trajectory[i];
    delete[] optimal_trajectory;
    /*
    if (data) {
	if (data->data.A) csc_spfree(data->data.A);
	if (data->data.P) csc_spfree(data->data.P);
	cleanup_rldl(data);
    }
    */
}

void VHMPC::initialize(int Nmax){
    this->Nmax = Nmax;
    // Allocate memory for all data dependent on Nmax
    linear_cost = new double[Nmax*(nx+nu)]();

    lb = new double[Nmax*(nx+ny)+nt];
    ub = new double[Nmax*(nx+ny)+nt];

    primal = new double[Nmax*(nx+nu)];
    dual = new double[Nmax*(nx+ny)+nt];


    optimal_trajectory = new double*[nx+nu];
    for (int i=0; i<nx+nu; i++)	optimal_trajectory[i] = new double[Nmax];
}

void VHMPC::cleanup_workspace(){
    if (work) osqp_cleanup(work);
    work = OSQP_NULL;
}

void VHMPC::cleanup_data(){
    if (data) {
	if (data->data.A) csc_spfree(data->data.A);
	if (data->data.P) csc_spfree(data->data.P);
	cleanup_rldl(data);
	data = OSQP_NULL;
    }
}

void VHMPC::copy_matrix(const Eigen::MatrixXd &M, csc *& N){
    int nz=0;
    for (int i=0;i<M.cols()*M.rows();i++)
        if (fabs(M.data()[i])>epsilon) nz++;
    N = csc_spalloc(M.rows(), M.cols(), nz, 1, 0);
    array_to_csc(N, M.data());
}


void VHMPC::load_constraints(){
    for (int i=0;i<Nmax;i++){
	for (int j=0;j<ny;j++){
	    lb[i*(nx+ny)+j] =   z_lower[j];
	    ub[i*(nx+ny)+j] =   z_upper[j];
	}
	for (int j=0;j<nx;j++){
	    lb[i*(nx+ny)+ny+j] =  0.0;
	    ub[i*(nx+ny)+ny+j] =  0.0;
	}
    }
    for (int j=0;j<nt;j++){
	    lb[Nmax*(nx+ny)+j] =  t_lower[j];
	    ub[Nmax*(nx+ny)+j] =  t_upper[j];
    }
    return;
}


void VHMPC::set_z_bound(const Eigen::MatrixXd &lower,const Eigen::MatrixXd &upper){
    for(int i=0;i<ny;i++){
	this->z_lower[i] = lower(i);
	this->z_upper[i] = upper(i);
    }
}
void VHMPC::set_terminal_bound(const Eigen::MatrixXd &lower,const Eigen::MatrixXd &upper){
    for(int i=0;i<nt;i++){
	this->t_lower[i] = lower(i);
	this->t_upper[i] = upper(i);
    }
}


int VHMPC::solve_N(const Eigen::MatrixXd &Ax0, int N1,
                   double * primal, double * dual, 
                   VHMPC_Result &results){
    update_time = 0.0;
    solve_time = 0.0;
    update_initial_constraint(Ax0);
    int solve_status =  solve_osqp(N1, primal, dual, results);
    save_solution(N1, update_time, solve_time, results);
    return solve_status;

}


double VHMPC::compute_cost(double obj_val, int N){
    return(obj_val  + 1.0*N);
}

void VHMPC::save_solution(int N, double update_time, double solve_time, VHMPC_Result &results){

	get_L_dimensions(work, &results.Ln, &results.Lm, &results.Lnnz);
	results.status_val = work->info->status_val;
	results.status_polish = work->info->status_polish;
	results.niter = work->info->iter;
        if (work->info->status_val == 1 || work->info->status_val == 2)
            results.objective_value = compute_cost(work->info->obj_val, N);
        else
            results.objective_value = -1;
	results.pri_res = work->info->pri_res;
	results.dua_res = work->info->dua_res;
        results.rho_estimate = work->info->rho_estimate;
        results.rho_updates = work->info->rho_updates;        
	results.osqp_setup_time = work->info->setup_time;
	results.osqp_solve_time = work->info->solve_time;
	results.osqp_update_time = work->info->update_time;
	results.osqp_polish_time = work->info->polish_time;
	results.solve_time = solve_time;
	results.update_time = update_time;
        results.horizon = N;        

        last_status = work->info->status_val;
    
	// Copy the primal and dual solution
	for (int i=0; i<N*(nx+nu); i++)    primal[i] = work->solution->x[i];
	for (int i=0; i<N*(nx+ny)+nt; i++)   dual[i] = work->solution->y[i];
        for (int i=0; i<N; i++){
            for (int j=0; j<nx+nu; j++){
                optimal_trajectory[j][i] = work->solution->x[i*(nx+nu) + j];
            }
        }
}

int VHMPC::solve_variable_horizon(const Eigen::MatrixXd &Ax0, int N1,
				double * primal,
				double * dual, 
				VHMPC_Result &results){
    solve_time = 0.0;
    update_time = 0.0;
    int status, dN=1; 
    double cmin, cnew;
    results.num_opt = 0;
    results.osqp_admm_2 = -1.0;
    results.osqp_admm_3 = -1.0;    


    int Ninit; // Previous solution horizon
    
    if (Nmin == -1) Ninit = Nprev;
    else{
        if (Nmin>1)  Ninit = Nmin-1;
        else Ninit = 1; // Cannot have a horizon lower than this
    }

    int Nnew = Ninit;
    Nmin = -1; // No optimal horizon computed
    
    update_initial_constraint(Ax0);
    
    // Solve with horizon N1
    compute_warm_start_shifted(N0, Nnew, primal, dual); // Warm start with previous solution
    solve_osqp(Nnew, primal, dual, results);
    results.num_opt++;
    results.osqp_admm_1 = solve_time;
    status = work->info->status_val;

    if (status != 1){ // Initial horizon unfeasible
	while(status != 1 && Nnew <= Nmax){
	    Nnew += dN;
	    if (settings->warm_start) update_warm_start(N0, Nnew, primal, dual);
	    solve_osqp(Nnew, primal, dual, results);
            results.num_opt++;
            status = work->info->status_val;
	}
	Nmin = Nnew;
	cmin = compute_cost(work->info->obj_val, Nnew);
	save_solution(Nnew, update_time, solve_time, results);

    }else{ // Initial horizon feasible
	Nmin = Nnew;	
	cmin = compute_cost(work->info->obj_val, Nnew);
	save_solution(Nnew, update_time, solve_time, results);
	
	// Try reducing the horizon
	Nnew = Ninit+1;
        if (settings->warm_start) update_warm_start(N0, Nnew, primal, dual);
	solve_osqp(Nnew, primal, dual, results);
        results.num_opt++;
        results.osqp_admm_2 = solve_time - results.osqp_admm_1;        
	status = work->info->status_val;
	cnew = compute_cost(work->info->obj_val, Nnew);

	if (status == 1 && cnew<=cmin) { // If decrease lowers cost
	    dN = +1;
	}else{
            if (Ninit<=10) {// Cannot reduce horizon more than this
                deltaN = Nmin - Ninit;
                return 0; 
            }
            
	    // Try increasing the horizon
	    Nnew = Ninit-1;
	    if (settings->warm_start) update_warm_start(N0, Nnew, primal, dual);
	    solve_osqp(Nnew, primal, dual, results);
            results.num_opt++;
            results.osqp_admm_3 = solve_time - results.osqp_admm_2 - results.osqp_admm_1;                    
            status = work->info->status_val;
	    cnew = compute_cost(work->info->obj_val, Nnew);
            
	    if (status == 1 && cnew<=cmin) { // If decrease lowers cost
		dN = -1;
	    }else{ // Already in optimal value
		Nmin = Ninit;
		deltaN = 0;
                results.solve_time = solve_time;
                results.update_time = update_time;
                
		return 0;
	    }
	}
        while (status == 1 && cnew <= cmin && Nnew < Nmax && Nnew > 2){
	    cmin = cnew;
	    Nmin = Nnew;
	    save_solution(Nnew, update_time, solve_time, results);

	    // Test the next value
	    Nnew += dN;
	    solve_osqp(Nnew, primal, dual, results);
            results.num_opt++;
            status = work->info->status_val;
	    cnew = compute_cost(work->info->obj_val, Nnew);
	}

    }
    deltaN = Nmin - Ninit;
    return 0;

}



nominal_MPC::nominal_MPC(int nx, int nu, int ny, int nt): VHMPC(nx, nu, ny, nt){

}

recursive_MPC::recursive_MPC(int nx, int nu, int ny, int nt): VHMPC(nx, nu, ny, nt){
}


combined_MPC::combined_MPC(int nx, int nu, int ny, int nt): VHMPC(nx, nu, ny, nt){
}




// TODO: Update directly in work->x and work->y
void VHMPC::compute_warm_start(int Nold, int Nnew, double *x, double *y){
    if (last_status == 1){
        // Copy previous x and y solution
        for (int j=0;j<Nold*(nx+nu);j++){
            x[j] = primal[j];
        }
        for (int j=0;j<Nold*(nx+ny);j++){
            y[j] = dual[j];
        }

	if (Nnew>Nold){ // Compute the rest of the solution
	    Eigen::MatrixXd next_state(nx,1);
	    Eigen::MatrixXd state(nx,1);
	    Eigen::MatrixXd input(nu,1);
	    for (int i=0;i<nu;i++)  input(i,0) = primal[(Nold-1)*(nx+nu)+i];
	    for (int i=0;i<nx;i++)  state(i,0) = primal[(Nold-1)*(nx+nu)+nu+i];
	    for (int k=Nold; k<Nnew;k++){
                input = K*state;
		next_state = A*state + B*input;

		for (int i=0;i<nu;i++)  x[k*(nx+nu)+i] = input(i);
		for (int i=0;i<nx;i++)  x[k*(nx+nu)+nu+i] = next_state(i);
                
                // Warm start the y variable with last value
		for (int i=0;i<nx+ny;i++)  y[k*(nx+ny)+i] = y[(Nold-1)*(nx+ny)+i];

		state = next_state;
	    }
	}
        // Copy the terminal y value
	for (int i=0;i<nt;i++)  y[Nnew*(nx+ny)+i] = dual[Nold*(nx+ny)+i];
        settings->warm_start = true;
	enable_warm_start();

    }else{
	settings->warm_start=false;
	disable_warm_start();
    }
}


// TODO: Update directly in work->x and work->y
void VHMPC::compute_warm_start_shifted(int Nold, int Nnew, double *x, double *y){
    if (last_status == 1){
        // Shift previous x and y solution: from 1->Nold to 0->Nold-1
        for (int j=0;j<(Nold-1)*(nx+nu);j++){
            x[j] = primal[j+(nx+nu)];
        }
        for (int j=0;j<(Nold-1)*(nx+ny);j++){
            y[j] = dual[j+(nx+ny)];
        }

	if (Nnew>=Nold){ // Compute the rest of the solution
	    Eigen::MatrixXd next_state(nx,1);
	    Eigen::MatrixXd state(nx,1);
	    Eigen::MatrixXd input(nu,1);
	    for (int i=0;i<nu;i++)  input(i,0) = primal[(Nold-1)*(nx+nu)+i];
	    for (int i=0;i<nx;i++)  state(i,0) = primal[(Nold-1)*(nx+nu)+nu+i];
	    for (int k=Nold-1; k<Nnew;k++){
                input = K*state;
		next_state = A*state + B*input;

		for (int i=0;i<nu;i++)  x[k*(nx+nu)+i] = input(i);
		for (int i=0;i<nx;i++)  x[k*(nx+nu)+nu+i] = next_state(i);
                
                // Warm start the y variable with last value
		for (int i=0;i<nx+ny;i++)  y[k*(nx+ny)+i] = y[(Nold-1)*(nx+ny)+i];

		state = next_state;
	    }
	}
        // Copy the terminal y value
	for (int i=0;i<nt;i++)  y[Nnew*(nx+ny)+i] = dual[Nold*(nx+ny)+i];

        settings->warm_start = true;
	enable_warm_start();

    }else{
	settings->warm_start=false;
	disable_warm_start();
    }
}


// If primal/dual has already been computed up until N=Nold
void VHMPC::update_warm_start(int Nold, int Nnew, double *x, double *y){
    if (Nnew>Nold){
	Eigen::MatrixXd next_state(nx,1);
	Eigen::MatrixXd state(nx,1);
	Eigen::MatrixXd input(nu,1);
	for (int i=0;i<nu;i++)  input(i,0) = x[(Nold-1)*(nx+nu)+i];
	for (int i=0;i<nx;i++)  state(i,0) = x[(Nold-1)*(nx+nu)+nu+i];
	for (int k=Nold; k<Nnew;k++){
	    input = K*state;
	    next_state = A*state + B*input;
	    
	    for (int i=0;i<nu;i++)  x[k*(nx+nu)+i] = input(i);
	    for (int i=0;i<nx;i++)  x[k*(nx+nu)+nu+i] = next_state(i);
                
	    // Warm start the y variable
	    for (int i=0;i<nx+ny;i++)  y[k*(nx+ny)+i] = y[(Nold-1)*(nx+ny)+i];

	    state = next_state;
	}

        // Copy the terminal y value
	for (int i=0;i<nt;i++)  y[Nnew*(nx+ny)+i] = dual[Nold*(nx+ny)+i];

        settings->warm_start = true;
	enable_warm_start();
    }else{
	// Should we do something when horizon is truncated?
    }
}




/*
 * Update constraints: Horizon changing from Nold to Nnew
 * l_osqp and u_osqp contains current constraints
 * x0 is the new initial constraint
 * by is the lower bound on output, bt_l,bt_u is the terminal constraints
 */
void VHMPC::update_initial_constraint(const Eigen::MatrixXd &x0){
    for (int i=0;i<nx;i++){
	lb[ny+i] = x0(i,0);
	ub[ny+i] = x0(i,0);
    }
}

void VHMPC::update_constraints(const int Nold, const int Nnew,
			     const Eigen::MatrixXd &x0){
    update_initial_constraint(x0);
    update_constraints(Nold, Nnew);

}
void VHMPC::update_constraints(const int Nold, const int Nnew){

    
    if (Nnew < Nold){ // If horizon is decreasing
	for (int i=0;i<ny;i++){
	    lb[Nnew*(nx+ny)+i] = z_lower[i];
	    ub[Nnew*(nx+ny)+i] = z_upper[i];
	}
    }else{ // If horizon is increased, fill up the rest
	for (int j=Nold;j<Nnew;j++){
	    for (int i=0;i<ny;i++){
		lb[j*(nx+ny)+i] = z_lower[i];
		ub[j*(nx+ny)+i] = z_upper[i];
	    }
	    for (int i=0;i<nx;i++){
		lb[j*(nx+ny)+ny+i] = 0.0;
		ub[j*(nx+ny)+ny+i] = 0.0;
	    }
	}

    }
    // Update terminal constraint
    for (int i=0;i<nt;i++){
	lb[Nnew*(nx+ny)+i] = t_lower[i];
	ub[Nnew*(nx+ny)+i] = t_upper[i];
    }
}




void VHMPC::compute_P_matrix(OSQPData * data, c_int N){
    int nvar = N*(nx+nu);
    int nconst = N*(nx+ny)+nt;


    data->P->n = nvar;
    data->P->m = nvar;

    int nz_P = 0;
    int P_row=0;

    // Initial constraint
    for (int i=0; i<Q0->n;i++){
        data->P->p[i] = nz_P;
        for (int j=Q0->p[i];j<Q0->p[i+1];j++){
            data->P->i[nz_P] = Q0->i[j];
            data->P->x[nz_P++] = Q0->x[j];
        }
    }
    P_row += Q0->m;

    for (int iter=0;iter<N-1;iter++){
        for (int i=0; i<Qi->n;i++){
            data->P->p[P_row+i] = nz_P;
            for (int j=Qi->p[i];j<Qi->p[i+1];j++){
                data->P->i[nz_P] = P_row + Qi->i[j];
                data->P->x[nz_P++] = Qi->x[j];
            }
        }
        P_row += Qi->m;
    }

    // Terminal constraint
    for (int i=0; i<QN->n;i++){
        data->P->p[P_row+i] = nz_P;
        for (int j=QN->p[i];j<QN->p[i+1];j++){
            data->P->i[nz_P] = P_row + QN->i[j];
            data->P->x[nz_P++] = QN->x[j];
        }
    }
    P_row += QN->m;
    data->P->p[P_row] = nz_P;

    return;
}

void VHMPC::compute_A_matrix(OSQPData * data, c_int N){
    int nvar = N*(nx+nu);
    int nconst = N*(nx+ny)+nt;


    // We allocate more than needed right now
    data->A->n = nvar;
    data->A->m = nconst;

    int nz_A = 0;
    int A_row=0, A_col=0;
    // Initial cost
    for (int i=0; i<A0->n;i++){
        data->A->p[i] = nz_A;
        for (int j=A0->p[i];j<A0->p[i+1];j++){
            data->A->i[nz_A] = A0->i[j];
            data->A->x[nz_A++] = A0->x[j];
        }
    }
    A_col += A0->n;

    for (int iter=0;iter<N-1;iter++){
        for (int i=0; i<Aij->n;i++){
            data->A->p[A_col+i] = nz_A;
            for (int j=Aij->p[i];j<Aij->p[i+1];j++){
                data->A->i[nz_A] = A_row + Aij->i[j];
                data->A->x[nz_A++] = Aij->x[j];
            }
            for (int j=Ai->p[i];j<Ai->p[i+1];j++){
                data->A->i[nz_A] = A_row + Aij->m + Ai->i[j];
                data->A->x[nz_A++] = Ai->x[j];
            }
        }
        A_row += Aij->m;
        A_col += Aij->n;
    }

    // Terminal cost
    for (int i=0; i<AN->n;i++){
        data->A->p[A_col+i] = nz_A;
        for (int j=Aij->p[i];j<Aij->p[i+1];j++){
            data->A->i[nz_A] = A_row + Aij->i[j];
            data->A->x[nz_A++] = Aij->x[j];
        }
        for (int j=AN->p[i];j<AN->p[i+1];j++){
            data->A->i[nz_A] = A_row + Aij->m + AN->i[j];
            data->A->x[nz_A++] = AN->x[j];
        }
    }
    A_row += AN->m;
    A_col += AN->n;
    data->A->p[A_col] = nz_A;

    return;
}




c_int VHMPC::update_AP_matrices(OSQPData * data, c_int Nold, c_int Nnew){
    
    if (Nnew == Nold) return 0;
    c_int nz_P, nz_A;
    c_int P_row, P_col;
    c_int A_row, A_col;
    c_int nvar =   Nnew*(nx+nu);
    c_int nconst = Nnew*(nx+ny)+nt;
    c_int Ntemp;

    data->P->n = nvar;
    data->P->m = nvar;
    data->A->n = nvar;
    data->A->m = nconst;

    if (Nnew < Nold) {  // Return last few columns
        Ntemp = Nnew -2;
        P_row = nu + Ntemp*(nx+nu);
        P_col = nu + Ntemp*(nx+nu);
        A_row = Ntemp*(nx+ny);
        A_col = nu + Ntemp*(nx+nu);
        nz_P = data->P->p[P_col];
        nz_A = data->A->p[A_col];
    }else{ // N is increasing
        Ntemp = Nold -1;
        P_row = nu + Ntemp*(nx+nu);
        P_col = nu + Ntemp*(nx+nu);
        A_row = Ntemp*(nx+ny);
        A_col = nu + Ntemp*(nx+nu);
        nz_P = data->P->p[P_col];
        nz_A = data->A->p[A_col];
    }

    for (c_int iter=Ntemp;iter<Nnew-1;iter++){
        for (c_int i=0; i<Qi->n;i++){                  // For column in Qi
            // Move column i to column P_row + i
            for (c_int j=Qi->p[i];j<Qi->p[i+1];j++){   // Iterate over all rows in Qi
		    data->P->i[nz_P] = P_row + Qi->i[j]; 
		    data->P->x[nz_P] = Qi->x[j];
		    nz_P++;
            }
            data->P->p[P_row+i+1] = nz_P; //  Pointer value for next is nz_P
        }
        P_row += Qi->m;

        for (c_int i=0; i<Aij->n;i++){
            data->A->p[A_col+i] = nz_A;
            for (c_int j=Aij->p[i];j<Aij->p[i+1];j++){
                data->A->i[nz_A] = A_row + Aij->i[j];
                data->A->x[nz_A++] = Aij->x[j];
            }
            for (c_int j=Ai->p[i];j<Ai->p[i+1];j++){
                data->A->i[nz_A] = A_row + Aij->m + Ai->i[j];
                data->A->x[nz_A++] = Ai->x[j];
            }
        }
        A_row += Aij->m;
        A_col += Aij->n;
    } // End iter
    data->P->n = nvar;
    // Add the terminal matrices
    for (c_int i=0; i<QN->n;i++){

        data->P->p[P_row+i] = nz_P;
        for (c_int j=QN->p[i];j<QN->p[i+1];j++){
	    data->P->i[nz_P] = P_row + QN->i[j];
	    data->P->x[nz_P++] = QN->x[j];
        }
    }
    P_row += QN->m;
    data->P->p[P_row] = nz_P;

    // Add the final AN->n columns
    for (c_int i=0; i<AN->n;i++){

        data->A->p[A_col+i] = nz_A;
        for (c_int j=Aij->p[i];j<Aij->p[i+1];j++){
            data->A->i[nz_A] = A_row + Aij->i[j];
            data->A->x[nz_A++] = Aij->x[j];
        }
        for (c_int j=AN->p[i];j<AN->p[i+1];j++){
            data->A->i[nz_A] = A_row + Aij->m + AN->i[j];
            data->A->x[nz_A++] = AN->x[j];
        }
    }

    A_row += AN->m;
    A_col += AN->n;

    data->A->p[A_col] = nz_A;

    return 0;


}



int nominal_MPC::setup(int N, int Nmax){
    std::chrono::steady_clock::time_point begin, end;
    Nmin = -1;
    Nprev = -1;
    deltaN = 0;
    last_status = -1;
    // Initialize data structure to zero
    data = (OSQPDataRLDL *)c_calloc(1, sizeof(OSQPDataRLDL));
    data->data.A = OSQP_NULL;
    data->data.P = OSQP_NULL;
    

    begin = std::chrono::steady_clock::now();
    
    int nmax = Nmax*(nu+nx);  //maximum number of variables
    data->data.n = N*(nu+nx);      //number of variables
    data->data.m = N*(ny+nx) + nt; //number of inequalities
    data->data.u = ub;
    data->data.l = lb;
    data->data.q = linear_cost;

    // Maximum number of nonzeros
    c_int A_nnz_max = Nmax*(Ai->p[Ai->n] + Aij->p[Aij->n]) + A0->p[A0->n] + AN->p[AN->n]; 
    data->data.A = csc_spalloc(data->data.m, nmax, A_nnz_max, 1, 0);
    data->data.A->n = data->data.n;
    // Maximum number of nonzeros
    c_int P_nnz_max = Nmax*Qi->p[Qi->n] + Q0->p[Q0->n] + QN->p[QN->n]; 
    data->data.P = csc_spalloc(data->data.n, nmax, P_nnz_max, 1, 0); 
    data->data.P->n = data->data.n;
    
    compute_A_matrix(&data->data, N);
    compute_P_matrix(&data->data, N);
     
    end = std::chrono::steady_clock::now();

    setup_time = (std::chrono::duration_cast<std::chrono::nanoseconds>
			   (end - begin).count())*1e-9;

    this->N0 = N; // Update current horizon
    this->Nprev = N; // Update current horizon
    return 0;
}

int nominal_MPC::update_osqp(int N1){
    update_constraints(Nprev, N1); 
    
    data->data.n = N1*(nu+nx);      //number of variables
    data->data.m = N1*(ny+nx) + nt; //number of inequalities
    data->data.u = ub;
    data->data.l = lb;
    data->data.q = linear_cost;
    compute_A_matrix(&data->data, N1);
    compute_P_matrix(&data->data, N1);

    //print_csc_matrix(data->data.P,"P0");
    /*
    update_AP_matrices(data->data, nx,  nu, ny,  nt,
		       Q0,  Qi, QN,
		       A0,  Ai, Aij, AN,
		       Nold, N1);
    */
    //print_csc_matrix(data->data.P,"P1");

    int return_val = osqp_setup(&work, &data->data, settings);
     
    return return_val;
}


/*
 * Solve the nominal MPC problem: ::so
 * N1 = The horizon
 * l_osqp, u_osqp: the (already updated) constraints
 * A0, Ai, Aij, AN: System constraint matrices
 * Q0, Qi, QN: System cost matrices
 */
int nominal_MPC::solve_osqp(int N1, 
			    double * warm_start_vector, double * warm_start_vector_dual, 
			    VHMPC_Result &results){
    int return_val = 0;

    
    std::chrono::steady_clock::time_point begin, end;
    begin = std::chrono::steady_clock::now();
    // Cleanup the existing workspace
    cleanup_workspace();
    return_val = update_osqp(N1);
    if (settings->warm_start) osqp_warm_start(work, warm_start_vector, warm_start_vector_dual);

    end = std::chrono::steady_clock::now();

    update_time += (std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count())*1e-9;
    
    if (return_val){
	printf("Error in setup of OSQP: %i\n", return_val);
	this->Nprev = -1; // Update current horizon
	this->N0 = -1; // Update current horizon
	return return_val;
	
    }else{
        begin = std::chrono::steady_clock::now();
	return_val = osqp_solve_recursive(work);
	end = std::chrono::steady_clock::now();
	        
	if (return_val){ 
	    printf("Error in solving normal MPC problem: %i\n", return_val);
	}else{
	    solve_time += (std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count())*1e-9;
	}
    }
    this->Nprev = N1; // Update current horizon
    this->N0 = N1; // Update current horizon
    return return_val;
}

/*
 * Setup the variable-horizon MPC. 
 * This only needs to be done once. 
 * l_osqp, u_osqp: the  constraints
 * A0, Ai, Aij, AN: System constraint matrices
 * Q0, Qi, QN: System cost matrices
 *
*/
int recursive_MPC::setup(int N, int Nmax){
    Nmin = -1;
    Nprev = -1;
    deltaN = 0;
    last_status = -1;
    
    // Initialize data structure to zero
    data = (OSQPDataRLDL *)c_calloc(1, sizeof(OSQPDataRLDL));
    data->data.A = OSQP_NULL;
    data->data.P = OSQP_NULL;
    
    std::chrono::steady_clock::time_point begin, end;
    begin = std::chrono::steady_clock::now();
    
    data->data.n = Nmax*(nu+nx);      // number of states
    data->data.m = Nmax*(ny+nx) + nt; // number of inequalities
    data->data.u = ub;
    data->data.l = lb;
    data->data.q = linear_cost;

    data->Nmax = Nmax;
    data->N = N;
    data->nx = nx;
    data->nu = nu;
    data->ny = ny;
    data->nt = nt;
    
    data->A0 = A0;
    data->Ai = Ai;
    data->Aij = Aij;
    data->AN = AN;
    data->Q0 = Q0;
    data->Qi = Qi;
    data->QN = QN;

    int return_val = osqp_setup_recursive(&work, data, settings, Nmax, N, nx, nu, ny, nt);
    end = std::chrono::steady_clock::now();
    setup_time = (std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count())*1e-9;
    
    if (return_val){
	printf("Error in setup of OSQP recursive: %i\n", return_val);
	return return_val;
    }
    
    this->Nprev = N; // Update current horizon
    this->N0 = N; // Update current horizon
    return 0;
}



/*
 * 
 * Solve the variable-horizon MPC (combined with standard OSQP factorization). 
 * l_osqp, u_osqp: the  constraints
 *
 */
int combined_MPC::solve_osqp(int N1, 
			double *primal, double *dual, 
			VHMPC_Result &results){
    std::chrono::steady_clock::time_point begin, end;
    // Start by updating the structure in OSQP
    begin = std::chrono::steady_clock::now();
    
    update_osqp(N1);
    if (settings->warm_start) osqp_warm_start(work, primal, dual);

    end = std::chrono::steady_clock::now();
    update_time += (std::chrono::duration_cast<std::chrono::nanoseconds>
			   (end - begin).count())*1e-9;
    
    
    begin = std::chrono::steady_clock::now();
    int return_val = osqp_solve_combine_recursive(work);
    end = std::chrono::steady_clock::now();

    if (return_val){
	printf("Error in solving variable MPC problem: %i\n", return_val);
	this->Nprev = -1; // Update current horizon
	this->N0 = -1; // Update current horizon	
        return return_val;
    }else{
	solve_time += (std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count())*1e-9;
	for (int i=0; i<N1; i++)
	    for (int j=0; j<nx+nu; j++)
	        optimal_trajectory[j][i] = work->solution->x[i*(nx+nu) + j];
    }

    this->Nprev = N1; // Update current horizon
    this->N0 = N1; // Update current horizon
    
    return 0;
}



/*
 * Setup the variable-horizon MPC which can be combined with the standard factorization. 
 *
 */
int combined_MPC::setup(int N, int Nmax){
    Nmin = -1;
    Nprev = -1;
    deltaN = 0;
    last_status = -1;
    std::chrono::steady_clock::time_point begin, end;
    // Initialize data structure to zero
    data = (OSQPDataRLDL *)c_calloc(1, sizeof(OSQPDataRLDL));
    data->data.A = OSQP_NULL;
    data->data.P = OSQP_NULL;
    
    begin = std::chrono::steady_clock::now();
    
    data->data.n = Nmax*(nu+nx);      // number of states
    data->data.m = Nmax*(ny+nx) + nt; // number of inequalities
    data->data.u = ub;
    data->data.l = lb;
    data->data.q = linear_cost;

    data->Nmax = Nmax;
    data->N = N-3;
    data->nx = nx;
    data->nu = nu;
    data->ny = ny;
    data->nt = nt;
    
    data->A0 = A0;
    data->Ai = Ai;
    data->Aij = Aij;
    data->AN = AN;
    data->Q0 = Q0;
    data->Qi = Qi;
    data->QN = QN;

    int return_val = osqp_setup_combine_recursive(&work, data, settings,
						  Nmax, N-1, 1,
						  nx, nu, ny, nt);
    end = std::chrono::steady_clock::now();
    setup_time = (std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count())*1e-9;
    
    if (return_val){
	printf("Error in setup of OSQP recursive: %i\n", return_val);
	return return_val;
    }
    
    this->Nprev = N; // Update current horizon
    this->N0 = N; // Update current horizon    
    return 0;
}


/*
 * Update the variable-horizon MPC after it has already been initialized.
 * N0 is the old horizon, N1 is the new. 
 * l_osqp, u_osqp: the updated constraints
 *
*/

int combined_MPC::update_osqp(int N1){
    int N0 = data->N;
    int return_val;

    update_constraints(N0, N1);
    

    // Nothing to update if the horizon has not changed

    if (N1 != N0)
	if (data->Nx > 0){
	    return_val = osqp_update_Z_horizon(work, data, N1-N0);
	}else
	    return_val = osqp_update_recursive(work, data, N1);
    else
	return_val = 0;
    
    // Update the initial value
    osqp_partial_update_bounds(work, ny, ny+nx, &lb[ny], &ub[ny]);
    
    // If increasing: Update state/input constraints
    if (N1 != N0){
	if (N1>N0){
	    for(int i=N0;i<N1;i++){
		osqp_partial_update_bounds(work, i*(nx+ny),  (i+1)*(nx+ny),
					   &lb[i*(nx+ny)], &ub[i*(nx+ny)]);
	    }
	}
	// Update the terminal constraint
    }
    osqp_partial_update_bounds(work, N1*(nx+ny),  N1*(nx+ny)+nt,
			       &lb[N1*(nx+ny)], &ub[N1*(nx+ny)]);

    return return_val;
}

/*
 * Update the variable-horizon MPC after it has already been initialized.
 * N0 is the old horizon, N1 is the new. 
 * l_osqp, u_osqp: the updated constraints
 *
*/

int recursive_MPC::update_osqp(int N1){
    assert(N0 == data->N);
    int return_val;

    // Nothing to update if the horizon has not changed
    update_constraints(N0, N1); 

    if (N1 != N0)
	if (data->Nx > 0){
	    return_val = osqp_update_Z_horizon(work, data, N1-N0);
	}else
	    return_val = osqp_update_recursive(work, data, N1);
    else
	return_val = 0;
    
    // Update the initial value
    osqp_partial_update_bounds(work, ny, ny+nx, &lb[ny], &ub[ny]);
    
    // If increasing: Update state/input constraints
    if (N1 != N0){
	if (N1>N0){
	    for(int i=N0;i<N1;i++){
		osqp_partial_update_bounds(work, i*(nx+ny),  (i+1)*(nx+ny),
					   &lb[i*(nx+ny)], &ub[i*(nx+ny)]);
	    }
	}
	// Update the terminal constraint
    }
    osqp_partial_update_bounds(work, N1*(nx+ny),  N1*(nx+ny)+nt,
			       &lb[N1*(nx+ny)], &ub[N1*(nx+ny)]);

    return return_val;
}


/*
 * Solve the variable-horizon MPC after it has already been initialized.
 * N0 is the old horizon, N1 is the new. 
 * l_osqp, u_osqp: the updated constraints
 *
 */
int recursive_MPC::solve_osqp(int N1,
			      double *primal, double *dual, 
			      VHMPC_Result &results){

    std::chrono::steady_clock::time_point begin, end;
    begin = std::chrono::steady_clock::now();

    update_osqp(N1);        
    if (settings->warm_start) osqp_warm_start(work, primal, dual);

    end = std::chrono::steady_clock::now();
    
    update_time += (std::chrono::duration_cast<std::chrono::nanoseconds>
			   (end - begin).count())*1e-9;

    
    begin = std::chrono::steady_clock::now();
    int return_val = osqp_solve_recursive(work);
    end = std::chrono::steady_clock::now();

    if (return_val){
	printf("Error in solving variable MPC problem: %i\n", return_val);
	this->Nprev = -1; // Update current horizon
	this->N0 = -1; // Update current horizon	
        return return_val;
    }else{
	solve_time += (std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count())*1e-9;
	for (int i=0; i<N1; i++)
	    for (int j=0; j<nx+nu; j++)
	        optimal_trajectory[j][i] = work->solution->x[i*(nx+nu) + j];
    }
    
    this->Nprev = N1; // Update current horizon
    this->N0 = N1; // Update current horizon    
    return 0;
}
