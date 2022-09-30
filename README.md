## A simple example of the VH-MPC algorithm. 
This code provides a simple example of the Variable Horizon MPC alorithm, as presented in the following papers: 

[Linnea Persson, Anders Hansson, and Bo Wahlberg. _A computationally fastvariable horizon MPC algorithm with application in rendezvous of autonomous unmanned vehicles._ ]()
  In (Submitted to IEEE Transactions on Control Systems Technology), June 2021.
  
[Linnea Persson and Bo Wahlberg. _Variable prediction horizon control forcooperative landing on moving target._](https://ieeexplore.ieee.org/document/9438459)
   In 2021 IEEE Aerospace Conference, March 2021.
   
Two methods are used to lower the total solve time when updating between different horizons: 
* The terminal cost and constraints are chosen so that the integer optimization in the VH-MPC problem becomes convex. Thus we only have to search for a local minima. 
* The similarities between the subproblems are exploited by deriving a recursive LDL factorization method. 

## Install
To run the code, you must first have the following version of OSQP installed: [https://github.com/laperss/osqp-recursive-ldl](https://github.com/laperss/osqp-recursive-ldl)

The code can be compiled using cmake
```sh
mkdir build
cd build
cmake ..
sudo make install

```

## Examples
The examples use a simple linear drone model, with 12 states and 4 inputs. 
### Increasing horizon demonstration
The files [VH_MPC_test_nominal.cpp](../blob/master/VH_MPC_test_nominal.cpp) and [VH_MPC_test_recursive.cpp](../blob/master/VH_MPC_test_recursive.cpp) 
contains the code demonstrating the time results of the nominal factorization and the recursive LDL factorization. 
The same problem (with the same initial condition) is solved for over an increasing horizon varying between N=40 and N=50. 
The average result over 20 runs is illustrated below: 

![Standard method](https://github.com/laperss/vh-mpc-example/blob/master/figures/compare.jpg)

### Increasing horizon demonstration
The files [VH_MPC_closed_loop_nominal.cpp](../blob/master/VH_MPC_closed_loop_nominal.cpp) and [VH_MPC_closed_loop_recursive.cpp](../blob/master/VH_MPC_closed_loop_recursive.cpp) 
demonstrates a closed-loop simulation where the system dynamics is the same as the dynamics in the MPC model. 
A comparison of the total solve time in each iteration is shown below. 

<img src="https://github.com/laperss/vh-mpc-example/blob/master/figures/closed_loop.jpg" width="500">
