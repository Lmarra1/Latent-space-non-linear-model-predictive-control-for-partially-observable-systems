"""
===============================================================================
File:         5_nmpc_latent_UKF.py
Author:       Luigi Marra
Email:        luigi.marra@uc3m.es

Description:  
    Implementation of Model Predictive Control in partial observability in the latent space from OpInf
    The estimation is performed using a UKF from sparse and noisy measurements data
    Implementation for the one dimensional Kuramoto-Sivashinsky equation

===============================================================================
"""



#%% Packages
# --------------------------------------------- #
import time, gc, os, h5py, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import general_utils.plant as plant_class
import general_utils.plant_model as plant_model_class
import general_utils.nmpc_latent_casadi as nmpc_class
import general_utils.ukf_class as UKF
from general_utils.utils import fourier_filtered_input
# --------------------------------------------- #
from general_utils.utils import psd
# --------------------------------------------- #



#%% Parameters of the plant

with h5py.File(os.path.join(parent_dir, 'data', '_KS', 'training_dataset.h5'), 'r') as f:
    uh_tr      = f['uh_tr'][:]
    xh_tr      = f['xh_tr'][:]
    t_tr       = f['t_tr'][:]
    
plant_type       = "KS" 
dt               = 0.1                                                         # Time interval between each action update (control timestep)
ny               = 4                                                           # Number of sensors in the domain
std_noise        = 0.3                                                         # Standard deviation of the measurement noise
n_warm_start_ukf = 1000                                                        # Warm start of the Kalman Filter in timesteps
sample_frequency = int(0.1/dt)                                                 # Interval between filter corrections
eq_target        = 3                                                           # ID of the control target. 0: E_0;    1: E_1;    2: E_2;    3: E_3;    other: time dependent trajectory     
save_data        = False                                                       # Bool to save control data
bool_plot        = True                                                        # Bool to plot control data


#%% Initialize the plant model for mpc and also save the reduced order model
model_path       = os.path.join(parent_dir, 'data', '_KS')
plant_model      = plant_model_class.plant(plant_type = plant_type, dt = dt, d_sparse = ny, model_path = model_path)
plant            = plant_class.plant(plant_type = plant_type, dt = plant_model.dt, d_sparse = ny)
rom              = plant_model.rom
basis            = rom.basis.entries
latent_dim       = rom.basis.reduced_state_dimension
ukf              = UKF.UKFStateEstimator(plant_model, ny, std_noise)

    
yh_tr = np.full((ny, xh_tr.shape[1]), np.nan)
for i in range(xh_tr.shape[1]):
        yh_tr[:, i] = np.interp(plant_model.csi_y_sparse, plant_model.csi, xh_tr[:, i])



#%% Read the control objective and compressed counterparts
with h5py.File('..//data//_KS//invariant_solutions.h5', 'r') as f:
    E0    = f['E0'][:]
    E1    = f['E1'][:]
    E2    = f['E2'][:]
    E3    = f['E3'][:]
    
q0 = rom.encode(E0)   
q1 = rom.encode(E1)
q2 = rom.encode(E2)
q3 = rom.encode(E3)

if eq_target == 0:
    Q  = np.column_stack([q1*0, q1*0])
    tq = [0,1]
elif eq_target == 1:
    Q  = np.column_stack([q1, q1])
    tq = [0,1]
elif eq_target == 2:
    Q = np.column_stack([q2, q2])
    tq = [0,1]
elif eq_target == 3:
    Q = np.column_stack([q3, q3])
    tq = [0,1]
else:
    Q = np.column_stack([q1, q2, q3])
    tq = [10,50,90]
    
q_ref_func = interp1d(
    tq,   
    Q,             
    axis         = 1,         
    kind         = 'linear',
    fill_value   = (Q[:, 0], Q[:, -1]),  # constant extrapolation at boundaries
    bounds_error = False
)



#%% Initialize nmpc controller
nmpc             = nmpc_class.nmpc(plant_model, q_ref_func)



#%% Set variable for the control application
n_episodes       = 100                                                         # Number of control trajectories
duration_episode = 25                                                          # Length of each trajectory
n_steps          = int(duration_episode/plant.dt)



#%%  Variable to store data during the control
xh               = np.full([n_episodes, plant.n,        n_steps + n_warm_start_ukf], np.nan)
yh               = np.full([n_episodes, plant.d_sparse, n_steps + n_warm_start_ukf], np.nan)
uh               = np.full([n_episodes, plant.m,        n_steps + n_warm_start_ukf], np.nan)
qh               = np.full([n_episodes, latent_dim,     n_steps + n_warm_start_ukf], np.nan)
qhath            = np.full([n_episodes, latent_dim,     n_steps + n_warm_start_ukf], np.nan)
xhath            = np.full([n_episodes, plant.n,        n_steps + n_warm_start_ukf], np.nan)
qstdh            = np.full([n_episodes, latent_dim,     n_steps + n_warm_start_ukf], np.nan)
xstdh            = np.full([n_episodes, plant.n,        n_steps + n_warm_start_ukf], np.nan)
coeffsh          = np.full([n_episodes, 3,              n_steps + n_warm_start_ukf], np.nan)
th               = np.arange(-n_warm_start_ukf*dt, duration_episode, step = plant.dt)
J_mpch           = np.full([n_episodes, n_steps], np.nan)
utm1             = np.zeros(plant_model.m)
traj_ref_h       = rom.decode(q_ref_func(th))
coeffs_traj      = psd(traj_ref_h, axis = 0)
norm_action      = 3
pre_ut_ukf       = np.full((plant.m, n_warm_start_ukf), np.nan)
pre_th           = np.arange(0,n_warm_start_ukf)*dt
for i in range(plant.m):
    action           = fourier_filtered_input(pre_th, cutoff_freq = 2) 
    pre_ut_ukf[i,:]  = norm_action * (action - np.mean(action, keepdims=True))/(np.std(action, keepdims=True) + 1e-8)



#%%  Read initial conditions from files
with h5py.File('..//data//_KS//initial_conditions.h5', 'r') as f:
    ICs    = f['IC'][:]
    


#%%  Start simulation...
print('--info:  Control simulation started...')
for _episode in range(n_episodes):
    ind              = 0
    utm1             = np.zeros(plant_model.m)
    ut               = np.zeros(plant_model.m)
    print(f'--info:  Episode {_episode} started...')
    
    start_ep     = time.time()
    x0           = ICs[:,_episode]
    q0           = rom.encode(x0)
    q_star       = q_ref_func(0)  # Read the target
    nmpc.refresh_actions()
    
    # Initialize the plant
    plant.set_IC(x0)
    plant.measure_state(std_noise)
    x_star = rom.decode(q_star)
    
    ukf.initialize_x_gpr(plant.y_sparse)
    # ukf.initialize_x_knn(xh_tr, yh_tr, plant.y_sparse)
    
    xh[_episode, :, ind]        = plant.x
    uh[_episode, :, ind]        = ut
    yh[_episode,:,ind]          = plant.y_sparse
    qh[_episode,:,ind]          = q0
    coeffsh[_episode,:,ind]     = psd(plant.x, 0)
    qhath[_episode,:,ind]       = ukf.qhat0
    xhath[_episode,:,ind]       = ukf.xhat0
    qstdh[_episode,:,ind]       = np.diag(ukf.ukf.P)
    xstdh[_episode,:,ind]       = np.diag(ukf.x0hat_cov_gpr)
    
    
    for _ws_ukf in range(n_warm_start_ukf):
        plant.advance(ut)
        plant.measure_state(std_noise)
        bool_update = ((_ws_ukf + 1) % sample_frequency == 0)
        qhat, qcov, xhat, xcov, _, _ = ukf.predict_correct_step(ut, plant.y_sparse, bool_update)
        print(f't = {((_ws_ukf + 1 - n_warm_start_ukf)*dt):.1f}')
        if bool_plot:
            if _ws_ukf % 100 == 0 and _ws_ukf != 0:
                hf = plt.figure(1, figsize=(8, 5))
                plt.clf()
                plt.plot(plant.csi, xhat, 'b--', linewidth=1, label = 'ukf estim. mean')
                plt.fill_between(plant.csi, xhat - np.sqrt(np.diag(xcov)), xhat + np.sqrt(np.diag(xcov)), color='blue', alpha=0.2, label = 'ukf estim. uncertainty (1 $\sigma$)')
                plt.plot(plant.csi, plant.x, 'k-',               linewidth=1, label = 'Ground truth')
                plt.plot(plant.csi_y_sparse, plant.y_sparse, 'o',             label = 'measure')
                plt.ylim(-7, 6)
                plt.xlim(0, 22)
                plt.legend(ncol=2, frameon=False, loc='lower center')
                plt.xlabel(r"$\xi$")
                plt.ylabel(r"$x(\xi, t)$")
                plt.title(rf'ep: {_episode} - warm start ukf - $t = {((_ws_ukf + 1 - n_warm_start_ukf)*dt):.1f}$')
                plt.show()
                plt.pause(0.01)
               
        ut = pre_ut_ukf[:,_ws_ukf]
        
        # Store data
        ind+=1
        xh[_episode, :, ind]        = plant.x
        yh[_episode,:,ind]          = plant.y_sparse
        uh[_episode,:,ind]          = ut
        qh[_episode,:,ind]          = rom.encode(plant.x)
        qhath[_episode,:,ind]       = qhat
        xhath[_episode,:,ind]       = xhat
        qstdh[_episode,:,ind]       = np.sqrt(np.diag(qcov))
        xstdh[_episode,:,ind]       = np.sqrt(np.diag(xcov))
        coeffsh[_episode,:,ind]     = psd(plant.x, 0)

    # Start the episode
    for _step in range(n_steps-1):
        start_st = time.time()
        
        ut, Jt       = nmpc.get_action(plant.t, qhat, q_ref_func)  
        
        # One-step plant integration and state measurement
        plant.advance(ut)
        plant.measure_state(std_noise)
        
        
        bool_update = ((_ws_ukf + _step + 1) % sample_frequency == 0)
        qhat,qcov,xhat,xcov,_,_ = ukf.predict_correct_step(ut, plant.y_sparse, bool_update)
        
        # Store episodes results
        ind+=1
        xh[_episode,:,ind]       = plant.x
        yh[_episode,:,ind]       = plant.y_sparse
        qh[_episode,:,ind]       = rom.encode(plant.x)
        uh[_episode,:,ind]       = ut
        
        coeffsh[_episode,:,ind]  = psd(plant.x, 0)
        qstdh[_episode,:,ind]    = np.sqrt(np.diag(qcov))
        xstdh[_episode,:,ind]    = np.sqrt(np.diag(xcov))
        qhath[_episode,:,ind]    = qhat
        xhath[_episode,:,ind]    = xhat
        
        J_mpch[_episode,_step]   = Jt
        end_st                   = time.time()
        el_time_st               = end_st - start_st
        utm1                     = ut.copy()
        
        # Show progress during control application 
        print(f"ep: {_episode} - t = {_step*plant.dt:>5.3f} - J_nmpc = {Jt:>7.3f}, ElTime = {el_time_st:>7.3f}")
        if bool_plot:
            if _step % 10 == 0:
                hf = plt.figure(2, figsize=(8, 5))
                plt.clf()
                plt.plot(plant.csi, plant.x, 'k-', label='true state', linewidth=1)
                plt.plot(plant.csi, x_star, 'r-', label='target', linewidth=1)
                plt.plot(plant.csi, xhat, 'b--', label='ukf estim. mean', linewidth=1)
                plt.fill_between(plant.csi, xhat - np.sqrt(np.diag(xcov)), xhat + np.sqrt(np.diag(xcov)), color='blue', alpha=0.2, label = 'ukf estim. uncertainty (1 $\sigma$)')
                plt.plot(plant.csi_y_sparse, plant.y_sparse, 'o', label='measure')
                plt.legend(ncol=2, frameon=False, loc='lower center')
                plt.ylim(-7, 6)
                plt.xlim(0, 22)
                plt.xlabel(r"$\xi$")
                plt.ylabel(r"$x(\xi, t)$")
                plt.title(rf'ep: {_episode} - mpc control: states - $t = {_step * dt:.1f}$')
                plt.show()
                plt.pause(0.01)
    gc.collect()        
    end_ep              = time.time()
    el_time_ep          = end_ep - start_ep
    
    print(f'--info:  Episode {_episode} ended. Elapsed time: {round(el_time_ep,2)} s')
print(f'--info:  Control simulation ended.')


# Save the data 
if save_data: 
    file     = f'control_partial_obs_sens{int(ny)}_noise{int(std_noise*10)}_samplfreq{sample_frequency}_E{eq_target}.h5'
    filename = os.path.join(parent_dir, 'data', '_KS', 'results', 'control', 'partial_obs', file)
    with h5py.File(filename, "w") as f:
        f.create_dataset("xh", data=xh, compression="gzip")
        f.create_dataset("uh", data=uh, compression="gzip")
        f.create_dataset("th", data=th, compression="gzip")
        f.create_dataset("yh", data=yh, compression="gzip")
        f.create_dataset("qh", data=qh, compression="gzip")
        f.create_dataset("coeffsh", data=coeffsh, compression="gzip")
        f.create_dataset("J_mpch", data=J_mpch, compression="gzip")
        f.create_dataset("traj_ref_h", data=traj_ref_h, compression="gzip")
        f.create_dataset("coeffs_traj", data=coeffs_traj, compression="gzip")
        f.create_dataset("qstdh", data=qstdh, compression="gzip")
        f.create_dataset("xstdh", data=xstdh, compression="gzip")
        f.create_dataset("qhath", data=qhath, compression="gzip")
        f.create_dataset("xhath", data=xhath, compression="gzip")
        f.create_dataset("ny", data=ny)
        f.create_dataset("std_noise", data=std_noise)
        f.create_dataset("sample_frequency", data=sample_frequency)
        f.create_dataset("n_warm_start_ukf", data=n_warm_start_ukf)
        



