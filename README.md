# CRBMG
CRBMG solves the (to be filled later). The equations are in **Rationalized-Lorentz-Heaviside-QCD Unit**.

## 1. The EMsolver Module
EMsolver implements the Jefimenko's equations on GPUs. 
If you use this module to perform electromagnetic calculations, please cite us via

(to be filled later)

> **To understand how EMsolver works, please refer to**

### 1.1 Installation
Installation via pip is encouraged. To run EMsolver, the following packages need to be pre-installed:
  - Numba
  - Ray
  - cupy
  - matplotlib
  - cudatoolkit

To start up, create a conda environment and install CRBMG:
```
# create a new environment
$: conda create -n CRBMG

# install relavant package sequentially
$: conda install numba
$: pip install -U ray
$: conda install cupy matplotlib
$: conda install matplotlib
$: pip install CRBMG
```
Execute the test file ---  **'test of EMsolver.ipynb'** in the repository before any real tasks.

### 1.2 Usage via an example
The following codes demonstrate an axample of how to use EMsolver.
```
from EMsolver.solver import EMsolver
import numpy as np
import math
import os
import cupy
from numba import cuda
import ray
from EMsolver.region_distance import signal_indicator
import matplotlib.pyplot as plt
from timeit import default_timer as timer

# start ray server
ray.init(ignore_reinit_error=True)
```
Suppose that the sources of charge density \rho and current density J are in region [[-3, 3],[-3, 3],[-3, 3]] GeV^-3, 
while the observational region of EM fields are in region [[-3, 3],[-3, 3],[10, 16]] GeV^-3. 
If we take 5 grids in each spatial direction, then we have
```
# the regions are seperated as the source region and the observation region
x_grid_size_o, y_grid_size_o, z_grid_size_o = 5,5,5
x_grid_size_s, y_grid_size_s, z_grid_size_s = 5,5,5

# the infinitesimals of the regions
# here the source region and observational region are overlap
dx_o, dy_o, dz_o, x_left_boundary_o, y_left_boundary_o, z_left_boundary_o = \
                       6/x_grid_size_o, 6/y_grid_size_o, 6/z_grid_size_o, -3, -3, -3
dx_s, dy_s, dz_s, x_left_boundary_s, y_left_boundary_s, z_left_boundary_s = \
                       6/x_grid_size_s, 6/y_grid_size_s, 6/z_grid_size_s, -3, -3, 10
```
Jefimenko's equations involve integrations of the retarded time. 
In this example, the longest distance between the source and observational regions is math.sqrt(6^2+6^2+19^2) ~ 20.81 GeV^-1.
Therefore, if we choose dt = 0.05 GeV^-1, the maximum length of time_snapshots should be 20.81/0.05 ~ 420.
This means that we only need to store 420 time steps of \rho and J in the GPU memory.
```
dt = 0.05
# define the length of the sources
# rho_GPU and Jx_GPU are of shape [len_time_snapshots, total_grid_size]
len_time_snapshots = 420
```
We also choose the GPU '0'
```
i_GPU = '0'
```
Now we load the remote class
```
# load the remote server and set up the constant sources of \rho and J
f = EMsolver.remote(len_time_snapshots, i_GPU, \
                    x_grid_size_o, y_grid_size_o, z_grid_size_o, \
                    x_grid_size_s, y_grid_size_s, z_grid_size_s, \
                    dx_o, dy_o, dz_o, x_left_boundary_o, y_left_boundary_o, z_left_boundary_o, \
                    dx_s, dy_s, dz_s, x_left_boundary_s, y_left_boundary_s, z_left_boundary_s, \
                    dt)
       
# toy model of constant sources
rho= np.ones(x_grid_size_s*y_grid_size_s*z_grid_size_s, dtype=np.float32)
Jx, Jy, Jz = (np.ones(x_grid_size_s*y_grid_size_s*z_grid_size_s, dtype=np.float32) for _ in range(3))       
```
If we only save the data of E and B in the XOY plane, the execution will be
```
# This is for saving zero E and B, can be neglected if not used.
Ex, Ey, Bx, By = (np.zeros(x_grid_size_s*y_grid_size_s*z_grid_size_s, dtype=np.float32) for _ in range(4))

# save the results in these lists
Ex_list, Ey_list, Ez_list, Bx_list, By_list, Bz_list= [], [], [], [], [], []

start = timer()
# run 410 time steps
for time in range(410):
  
    # updata new rho and J 
    f.update_rho_J.remote(rho, Jx, Jy, Jz)

    # make sure if the signal has been transmitted to the observaional region
    retarded_time = signal_indicator(dx_o, dy_o, dz_o, x_left_boundary_o, y_left_boundary_o, z_left_boundary_o, \
                                     dx_s, dy_s, dz_s, x_left_boundary_s, y_left_boundary_s, z_left_boundary_s, \
                                     x_grid_size_o, y_grid_size_o, z_grid_size_o, \
                                     x_grid_size_s, y_grid_size_s, z_grid_size_s)
    if time*dt >= retarded_time:
        Ex, Ey, _, Bx, By, _ = ray.get(f.Jefimenko_solver.remote())
        
        if time%1==0:
            Ex_list.append(Ex.reshape([x_grid_size_o, y_grid_size_o, z_grid_size_o])[:,:,2])
            Ey_list.append(Ey.reshape([x_grid_size_o, y_grid_size_o, z_grid_size_o])[:,:,2])
            Bx_list.append(Bx.reshape([x_grid_size_o, y_grid_size_o, z_grid_size_o])[:,:,2])
            By_list.append(By.reshape([x_grid_size_o, y_grid_size_o, z_grid_size_o])[:,:,2])
end = timer()
print('evaluation time:',end-start)   
```
> the calculated result of Ex_list[120] will be:
```
[[-8.5784625e-03, -9.4471080e-03, -9.8685343e-03, -9.4471080e-03, -8.5784625e-03],
 [-4.6509719e-03, -5.0742337e-03, -5.1472820e-03, -5.0742328e-03, -4.6509719e-03],
 [ 2.4012384e-10,  0.0000000e+00, -9.6049529e-11, -2.4012384e-10, -2.2411557e-10],
 [ 4.6509723e-03,  5.0742342e-03,  5.1472820e-03,  5.0742342e-03,  4.6509723e-03],
 [ 8.5784635e-03,  9.4471108e-03,  9.8685334e-03,  9.4471090e-03,  8.5784625e-03]]
```

## License
The package is coded by Jun-Jie Zhang.

This package is free you can redistribute it and/or modify it under the terms of the Apache License Version 2.0, January 2004 (http://www.apache.org/licenses/).

For further questions and technical issues, please contact us at

zjacob@mail.ustc.edu.cn (Jun-Jie Zhang 张俊杰)

**File Structure**
```
CRBMG
│   README.md 
│   LICENSE
│   setup.py 
│   test of EMsolver.ipynb
│
└───EMsolver
    │   cuda_functions.py
    │   region_distance.py
    │   slover.py
    │   __init__.py
```
