"""
===============================================================================
File:         5_nmpc_latent_UKF.py
Author:       Luigi Marra
Email:        luigi.marra@uc3m.es

Description:  
    Implementation of Model Predictive Control in partial observability in the latent space from OpInf
    The estimation is performed using a UKF from sparse and noisy measurements data
    Implementation for the two dimensional Kuramoto-Sivashinsky equation

===============================================================================
"""



#%% Packages
# --------------------------------------------- #
import time, gc, h5py, os, sys, dill
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import general_utils.plant as plant_class
import general_utils.plant_model as plant_model_class
import general_utils.nmpc_latent_casadi_ks2d as nmpc_class
import general_utils.ukf_class_2D as UKF
from general_utils.utils import fourier_filtered_input
from matplotlib import colors
# --------------------------------------------- #

flat_red     = colors.ListedColormap(['red'])
flat_blue    = colors.ListedColormap(['blue'])
flat_green   = colors.ListedColormap(['green'])



#%% Parameters of the plant
dataset          = 1
model_path       = f'..//data//_KS2D//dataset_{dataset}'
with open(os.path.join(model_path, 'rom.dill'), 'rb') as f:
    rom = dill.load(f)
with h5py.File(os.path.join(model_path, 'params.h5'), 'r') as f:
    dt    = f['dtrom'][()];

plant_type       = "KS2D" 
std_noise        = 0.3                                                         # Standard deviation  of the measurement noise
sample_frequency = int(0.1/dt)                                                 # Time interval between filter corrections
n_sens_grid      = 5                                                           # Number of sensors along one side of the grid: n_sens_grid x n_sens_grid
n_warm_start_ukf = int(10/dt)                                                  # Number of timesteps for filter warm start

save_data        = False                                                       # Bool variable to save control results
bool_plot        = True                                                        # Bool variable to plot control results



plant_model      = plant_model_class.plant(plant_type = plant_type, dt = dt, model_path = model_path, n_sens_grid = n_sens_grid)
plant            = plant_class.plant(plant_type = plant_type, dt = plant_model.dt, n_sens_grid = n_sens_grid)
latent_dim       = rom.basis.reduced_state_dimension
rom              = plant_model.rom
basis            = rom.basis.entries

#%% Control target
with h5py.File(f'..//data//_KS2D//dataset_{dataset}//target.h5', 'r') as f:
    x_star      = f['x_star'][:]  
    
    
q_star           = rom.encode(x_star)
x_star_hat       = rom.decode(q_star)
q_ref_func       = interp1d(np.arange(2, step = 1), np.tile(q_star.reshape(-1,1), 2), kind='nearest', fill_value='extrapolate', bounds_error=False)

csi_x            = plant_model.csi_x
csi_y            = plant_model.csi_y
csi_x_grid       = plant_model.csi_x_grid
csi_y_grid       = plant_model.csi_y_grid
csi_x_sens_grid  = plant_model.csi_x_sens_grid
csi_y_sens_grid  = plant_model.csi_y_sens_grid
ny               = np.prod(csi_y_sens_grid.shape)
ukf              = UKF.UKFStateEstimator(plant_model, ny, std_noise)
basis            = rom.basis.entries



#%% Initialize nmpc controller
nmpc             = nmpc_class.nmpc(plant_model, q_ref_func)



#%% Set variable for the control application
n_episodes       = 20
duration_episode = 10
n_steps          = int(duration_episode/plant.dt)



#%% Variable to store data during the control
xh               = np.full([n_episodes, plant.n,        n_steps + n_warm_start_ukf], np.nan)
yh               = np.full([n_episodes, plant.d_sparse, n_steps + n_warm_start_ukf], np.nan)
uh               = np.full([n_episodes, plant.m,        n_steps + n_warm_start_ukf], np.nan)
qh               = np.full([n_episodes, latent_dim,     n_steps + n_warm_start_ukf], np.nan)
qhath            = np.full([n_episodes, latent_dim,     n_steps + n_warm_start_ukf], np.nan)
xhath            = np.full([n_episodes, plant.n,        n_steps + n_warm_start_ukf], np.nan)
qstdh            = np.full([n_episodes, latent_dim,     n_steps + n_warm_start_ukf], np.nan)
xstdh            = np.full([n_episodes, plant.n,        n_steps + n_warm_start_ukf], np.nan)
th               = np.arange(-n_warm_start_ukf*dt, duration_episode, step = plant.dt)
J_mpch           = np.full([n_episodes, n_steps], np.nan)
rewh             = np.full([n_episodes, n_steps], np.nan)
u0               = np.zeros(plant_model.m)
utm1             = u0.copy()


norm_action      = 3
pre_ut_ukf       = np.full((plant.m, n_warm_start_ukf), np.nan)
pre_th           = np.arange(0,n_warm_start_ukf)*dt
for i in range(plant.m):
    action           = fourier_filtered_input(pre_th, cutoff_freq = 1) 
    pre_ut_ukf[i,:]  = norm_action * (action - np.mean(action, keepdims=True))/(np.std(action, keepdims=True) + 1e-8)


# Read initial condition for control 
with h5py.File(os.path.join(model_path, 'initial_conditions.h5'), 'r') as f:
    ICs    = f['IC'][:]

#%% Start simulation...

print('--info:  Control simulation started...')
for _episode in range(n_episodes):
    ind              = 0
    ut               = np.zeros(plant_model.m)
    utm1             = np.zeros(plant_model.m)
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
    
    xh[_episode, :, ind]        = plant.x
    uh[_episode, :, ind]        = ut
    yh[_episode,:,ind]          = plant.y_sparse
    qh[_episode,:,ind]          = q0
    qhath[_episode,:,ind]       = ukf.qhat0
    xhath[_episode,:,ind]       = ukf.xhat0
    qstdh[_episode,:,ind]       = np.diag(ukf.ukf.P)
    xstdh[_episode,:,ind]       = np.diag(ukf.x0hat_cov_gpr)
    
    for _ws_ukf in range(n_warm_start_ukf):
        plant.advance(ut)
        plant.measure_state(std_noise)
        bool_update = ((_ws_ukf + 1) % sample_frequency == 0)
        qhat, qcov, xhat, xcov, _, _ = ukf.predict_correct_step(ut, plant.y_sparse, bool_update)
        
        
        if bool_plot:
            if _ws_ukf % int(1/dt) == 0 and _ws_ukf != 0:
                print(f't = {((_ws_ukf + 1 - n_warm_start_ukf)*dt):.1f}')
                hf    = plt.figure(2,figsize=(6, 5))
                plt.clf()
                ax3d  = hf.add_subplot(111, projection='3d')
                downs = 1
                X     = plant_model.csi_x_grid[::downs, ::downs]
                Y     = plant_model.csi_y_grid[::downs, ::downs]
                Z     = xhat.reshape(plant_model.csi_x_grid.shape)[::downs, ::downs]
                Z2    = plant.x.reshape(plant_model.csi_x_grid.shape)[::downs, ::downs]  
                surf  = ax3d.plot_surface(X, Y, Z, cmap=flat_green, edgecolor='k', linewidth=0.2, alpha=0.5, label = 'UKF estim. mean')
                surf2 = ax3d.plot_surface(X, Y, Z2, cmap=flat_blue, edgecolor='k', linewidth=0.2, alpha=0.5, label = 'Ground truth')
                ax3d.set_xlabel(r"$\xi$")
                ax3d.set_ylabel(r"$\eta$")
                ax3d.set_zlabel(r"$x(\xi, \eta, t)$")
                ax3d.set_title(rf'ep: {_episode} - warm start ukf - $t = {((_ws_ukf + 1 - n_warm_start_ukf)*dt):.1f}$')
                ax3d.legend(loc='lower left')
                plt.tight_layout()
                plt.show()
                plt.pause(0.1) 
                
                
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
    
    # Start the episode
    for _step in range(n_steps-1):
        start_st = time.time()
        
        ut, Jt       = nmpc.get_action(plant.t, qhat, q_ref_func)  
        
        # One-step plant integration and state measurement
        plant.advance(ut)
        plant.measure_state(std_noise)
        
        bool_update = ((_ws_ukf + _step + 1) % sample_frequency == 0)
        qhat,qcov,xhat,xcov,_,_ = ukf.predict_correct_step(ut, plant.y_sparse, bool_update)
        
        # Calculate the reward
        rew_s         = -((plant.x - x_star).T @ nmpc.Qx @ (plant.x - x_star)).copy()
        
        
        # Store episodes results
        ind+=1
        xh[_episode,:,ind]       = plant.x
        yh[_episode,:,ind]       = plant.y_sparse
        qh[_episode,:,ind]       = rom.encode(plant.x)
        uh[_episode,:,ind]       = ut
        
        qstdh[_episode,:,ind]    = np.sqrt(np.diag(qcov))
        xstdh[_episode,:,ind]    = np.sqrt(np.diag(xcov))
        qhath[_episode,:,ind]    = qhat
        xhath[_episode,:,ind]    = xhat
        
        J_mpch[_episode,_step]   = Jt
        end_st                   = time.time()
        el_time_st               = end_st - start_st
        utm1                     = ut.copy()
        
        
        # Show progress during control application 
        print(f"ep: {_episode} - t = {_step*plant.dt:>5.2f} - {'rew_s':<5}= {rew_s:>7.2f} - J_nmpc = {Jt:>7.1f}, ElTime = {el_time_st:>7.3f}")
        
        if bool_plot:
            if _step % int(0.06/dt) == 0:
                hf    = plt.figure(2,figsize=(6, 5))
                plt.clf()
                ax3d  = hf.add_subplot(111, projection='3d')
                downs = 1
                X     = plant_model.csi_x_grid[::downs, ::downs]
                Y     = plant_model.csi_y_grid[::downs, ::downs]
                Z     = plant.x.reshape(plant_model.csi_x_grid.shape)[::downs, ::downs]
                Z2    = rom.decode(q_ref_func(0)).reshape(plant_model.csi_x_grid.shape)[::downs, ::downs]  
                Z3    = xhat.reshape(plant_model.csi_x_grid.shape)[::downs, ::downs]
                surf  = ax3d.plot_surface(X, Y, Z, cmap=flat_blue, edgecolor='k', linewidth=0.2, alpha=0.5, label='true state')
                surf2 = ax3d.plot_surface(X, Y, Z2, cmap=flat_red, edgecolor='k', linewidth=0.2, alpha=0.5, label='target')
                surf  = ax3d.plot_surface(X, Y, Z3, cmap=flat_green, edgecolor='k', linewidth=0.2, alpha=0.5, label='ukf estim. mean')
                ax3d.set_title(rf'ep: {_episode} - mpc control:states - $t_j = {_step * dt:.3f}$')
                ax3d.set_xlabel(r"$\xi$")
                ax3d.set_ylabel(r"$\eta$")
                ax3d.set_zlabel(r"$x(\xi, \eta, t)$")
                plt.tight_layout()
                plt.draw()
                plt.show()
                plt.pause(0.1) 
                
                
                plt.figure(3, figsize=(8, 5))
                plt.clf()
                plt.plot(th[n_warm_start_ukf:_step+n_warm_start_ukf], uh[_episode,:,:_step].T, linewidth=1)
                plt.ylim(-10, 10)
                plt.xlabel(r"$t$")
                plt.ylabel(r"$u(t)$")
                plt.title(rf'ep: {_episode} - mpc control - $t = {_step * dt:.3f}$')
                plt.draw()
                plt.show()
                plt.pause(0.1)
    gc.collect()        
    end_ep              = time.time()
    el_time_ep          = end_ep - start_ep
    

    
    print(f'--info:  Episode {_episode} ended. Elapsed time: {round(el_time_ep,2)} s')
print('--info:  Control simulation ended.')




#%% Save the data 
if save_data:
    filename = f"..//data//_KS2D//dataset_{dataset}//results//control//partial_obs//control_partial_obs_sens{int(ny)}_noise{int(std_noise*10)}_samplfreq{sample_frequency}.h5"
    with h5py.File(filename, "w") as f:
        f.create_dataset("xh", data=xh, compression="gzip")
        f.create_dataset("uh", data=uh, compression="gzip")
        f.create_dataset("th", data=th, compression="gzip")
        f.create_dataset("yh", data=yh, compression="gzip")
        f.create_dataset("qh", data=qh, compression="gzip")
        f.create_dataset("J_mpch", data=J_mpch, compression="gzip")
        f.create_dataset("qstdh", data=qstdh, compression="gzip")
        f.create_dataset("xstdh", data=xstdh, compression="gzip")
        f.create_dataset("qhath", data=qhath, compression="gzip")
        f.create_dataset("xhath", data=xhath, compression="gzip")
        f.create_dataset("ny", data=ny)
        f.create_dataset("std_noise", data=std_noise)
        f.create_dataset("sample_frequency", data=sample_frequency)
        f.create_dataset("n_warm_start_ukf", data=n_warm_start_ukf)
        f.create_dataset("x_star", data=x_star, compression="gzip")
        
        
    

