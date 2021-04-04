import cupy
from numba import cuda
import math

# give the distance between two grids
@cuda.jit(device=True)
def r_distance(x1, y1, z1, x2, y2, z2):
    return math.sqrt((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)

# give the index of the retarded time from time_snapshots
@cuda.jit(device=True)
def retarded_time_index(t, dr, dt, i_round, len_time_snapshots):
    tr = t - dr
    
    # if the signal has not been transmitted
    if tr < 0.:
        return -1
    
    # the current time tr corresponds to the ti-th row
    ti = int(math.floor(tr/dt)%len_time_snapshots)

    # if ti corresponds to the last row
    if ti == len_time_snapshots - 1:
        return 0
    else:
        return ti + 1


# copy newly calculated rho and J to GPU memory
# This is the cuda kernel function
@cuda.jit
def copy_new_rho_J_to_GPU(i_time, rho, Jx, Jy, Jz, \
                          rho_GPU, Jx_GPU, Jy_GPU, Jz_GPU, \
                          total_grid_s):

    i = cuda.grid(1)

    # loop through each spatial grid
    if i < total_grid_s:
        rho_GPU[i_time, i] = rho[i]
        Jx_GPU[i_time, i] = Jx[i]
        Jy_GPU[i_time, i] = Jy[i]
        Jz_GPU[i_time, i] = Jz[i]
    
# give the real position from index and left_boundary
@cuda.jit(device=True)
def real_position(i, d, left_boundary):
    return i*d + d/2 + left_boundary

# cuda kernel for obtaining E(t)  and B(t)  with given rho(t) /Jx(t) /Jy(t) /Jz(t) 
# time_snapshots only saves the recent time info, a full time info is on cpu and denoted as time_sequence
# rho_GPU, Jx_GPU, Jy_GPU, Jz_GPU are of shape [len_time_snapshots, total_grid_s]
@cuda.jit
def Jefimenko_kernel(rho_GPU_s, Jx_GPU_s, Jy_GPU_s, Jz_GPU_s, \
                     len_time_snapshots, i_round, i_time, dt, t, \
                     total_grid_o, total_grid_s, \
                     x_grid_size_o, y_grid_size_o, z_grid_size_o, \
                     x_grid_size_s, y_grid_size_s, z_grid_size_s, \
                     dx_o, dy_o, dz_o, \
                     dx_s, dy_s, dz_s, \
                     x_left_boundary_o, y_left_boundary_o, z_left_boundary_o, \
                     x_left_boundary_s, y_left_boundary_s, z_left_boundary_s, \
                     GEx, GEy, GEz, GBx, GBy, GBz):
    
    # the i-th spatial grid
    i_o = cuda.grid(1)
    
    # loop through each spatial grid in the observation region
    if i_o < total_grid_o:
        # convert 1D grid index into 3D
        iz_o = i_o%z_grid_size_o
        iz_rest_o = i_o//z_grid_size_o
        iy_o = iz_rest_o%y_grid_size_o
        iy_rest_o = iz_rest_o//y_grid_size_o
        ix_o = iy_rest_o%x_grid_size_o
        
        # express the position corresponding to the index
        x_o, y_o, z_o = real_position(ix_o, dx_o, x_left_boundary_o), \
                        real_position(iy_o, dy_o, y_left_boundary_o), \
                        real_position(iz_o, dz_o, z_left_boundary_o)
      
        # loop through all spatial grid in the source region for integration
        for i_s in range(total_grid_s):
            # 1D to 3D
            iz_s = i_s%z_grid_size_s
            iz_rest_s = i_s//z_grid_size_s
            iy_s = iz_rest_s%y_grid_size_s
            iy_rest_s = iz_rest_s//y_grid_size_s
            ix_s = iy_rest_s%x_grid_size_s
            # positions
            x_s, y_s, z_s = real_position(ix_s, dx_s, x_left_boundary_s), \
                            real_position(iy_s, dy_s, y_left_boundary_s), \
                            real_position(iz_s, dz_s, z_left_boundary_s)

            # dr = r - r' = r_o - r_s
            dx = x_o - x_s
            dy = y_o - y_s
            dz = z_o - z_s
            dr = r_distance(x_o, y_o, z_o, x_s, y_s, z_s)
            
            # exclude the divergence (this is the screening)
            if dr > 10**-10:                            
                # find the retarded time for positions r_o
                # rtime_i gives the index of time in time_snapshots, it can be negtive
                rtime_i = retarded_time_index(t, dr, dt, i_round, len_time_snapshots)

                # rtime_i_minus: index for retarded_time - dt
                if rtime_i < 0:
                    # no updation since signal has not arrived yet
                    pass
                else:
                    if rtime_i == 0:
                        rtime_i_minus = len_time_snapshots-1
                    elif rtime_i > 0:
                        rtime_i_minus = rtime_i - 1
                    
                    # partial_rho/partial_t, partial_Jx/partial_t, partial_Jy/partial_t, partial_Jz/partial_t
                    rho_s, rho_minus_s = rho_GPU_s[rtime_i, i_s], rho_GPU_s[rtime_i_minus, i_s]
                    Jx_s, Jx_minus_s = Jx_GPU_s[rtime_i, i_s], Jx_GPU_s[rtime_i_minus, i_s]
                    Jy_s, Jy_minus_s = Jy_GPU_s[rtime_i, i_s], Jy_GPU_s[rtime_i_minus, i_s]
                    Jz_s, Jz_minus_s = Jz_GPU_s[rtime_i, i_s], Jz_GPU_s[rtime_i_minus, i_s]
                    prho_pt_s = (rho_s - rho_minus_s)/dt
                    pJx_pt_s = (Jx_s - Jx_minus_s)/dt
                    pJy_pt_s = (Jy_s - Jy_minus_s)/dt
                    pJz_pt_s = (Jz_s - Jz_minus_s)/dt
                   
                    # Jefimenko solution
                    cuda.atomic.add(GEx, i_o, (x_o-x_s)/dr**3*rho_s+(x_o-x_s)/dr**2*prho_pt_s-1/dr*pJx_pt_s)
                    cuda.atomic.add(GEy, i_o, (y_o-y_s)/dr**3*rho_s+(y_o-y_s)/dr**2*prho_pt_s-1/dr*pJy_pt_s)
                    cuda.atomic.add(GEz, i_o, (z_o-z_s)/dr**3*rho_s+(z_o-z_s)/dr**2*prho_pt_s-1/dr*pJz_pt_s)
                    cuda.atomic.add(GBx, i_o, -(((y_o-y_s)*Jz_s-(z_o-z_s)*Jy_s)/dr**3+((y_o-y_s)*pJz_pt_s-(z_o-z_s)*pJy_pt_s)/dr**2))
                    cuda.atomic.add(GBy, i_o, -(((z_o-z_s)*Jx_s-(x_o-x_s)*Jz_s)/dr**3+((z_o-z_s)*pJx_pt_s-(x_o-x_s)*pJz_pt_s)/dr**2))
                    cuda.atomic.add(GBz, i_o, -(((x_o-x_s)*Jy_s-(y_o-y_s)*Jx_s)/dr**3+((x_o-x_s)*pJy_pt_s-(y_o-y_s)*pJx_pt_s)/dr**2)) 
