"""
===============================================================================
File:         4_nmpc_latent_full_observable.py
Author:       Luigi Marra
Email:        luigi.marra@uc3m.es

Description:  
    Implementation of Model Predictive Control in full observability in the latent space from OpInf
    Implementation for the two dimensional Kuramoto-Sivashinsky equation

===============================================================================
"""



#%% Packages
# --------------------------------------------- #
import time, gc, os, h5py, sys, dill
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import general_utils.plant as plant_class
import general_utils.plant_model as plant_model_class
import general_utils.nmpc_latent_casadi_ks2d as nmpc_class
# --------------------------------------------- #

flat_red     = colors.ListedColormap(['red'])
flat_blue    = colors.ListedColormap(['blue'])



#%% Parameters of the plant
plant_type       = "KS2D"                                                       
std_noise        = 0.0                                                         
dataset_number   = 1
model_path       = f'..//data//_KS2D//dataset_{dataset_number}'
with open(os.path.join(model_path, 'rom.dill'), 'rb') as f:
    rom = dill.load(f)
    
with h5py.File(os.path.join(model_path, 'params.h5'), 'r') as f:
    dt    = f['dtrom'][()];
    
    
plant_model      = plant_model_class.plant(plant_type = plant_type, dt = dt, model_path = model_path)
plant            = plant_class.plant(plant_type = plant_type, dt = plant_model.dt)
latent_dim       = rom.basis.reduced_state_dimension



#%% Control target
with h5py.File(os.path.join(model_path, 'target.h5'), 'r') as f:
    x_star      = f['x_star'][:]  

q_star           = rom.encode(x_star)
x_star_hat       = rom.decode(q_star)
q_ref_func       = interp1d(np.arange(2, step = 1), np.tile(q_star.reshape(-1,1), 2), kind='nearest', fill_value='extrapolate', bounds_error=False)



#%% Initialize nmpc controller
nmpc             = nmpc_class.nmpc(plant_model, q_ref_func)



#%% Set variable for the control application
n_episodes       = 100                                                         # Number of control trajectories
duration_episode = 15                                                          # Length of each trajectory
save_data        = False                                                       # Bool variable to save control results
n_steps          = int(duration_episode/plant.dt)



#%% Variable to store data during the control
xh               = np.full([n_episodes, plant.n,        n_steps], np.nan)
yh               = np.full([n_episodes, plant.d_sparse, n_steps], np.nan)
rewh             = np.full([n_episodes,                 n_steps], np.nan)
uh               = np.full([n_episodes, plant.m,        n_steps], np.nan)
qh               = np.full([n_episodes, latent_dim,     n_steps], np.nan)
th               = np.arange(0, n_steps, step = plant.dt)
J_mpch           = np.full([n_episodes, n_steps], np.nan)
rewh             = np.full([n_episodes, n_steps], np.nan)
utm1             = np.zeros(plant_model.m)

# Read initial condition for control 
with h5py.File(os.path.join(model_path, 'initial_conditions.h5'), 'r') as f:
    ICs    = f['IC'][:]
            

#%% Start simulation...
print('--info:  Control simulation started...')
for _episode in range(n_episodes):
    
    print(f'--info:  Episode {_episode} started...')
    start_ep     = time.time()
    x0           = ICs[:,_episode]
    
    q_star       = q_ref_func(0)  # Read the target
    nmpc.refresh_actions()
    
    # Initialize the plant
    plant.set_IC(x0)
    plant.measure_state(std_noise)
    
    # Store initial data
    xh[_episode,:,0]          = plant.x
    yh[_episode,:,0]          = plant.y_sparse
    rewh[_episode,0]          = -((plant.x - x_star).T @ nmpc.Qx @ (plant.x - x_star)).copy()

    
    # Start the episode
    for _step in range(n_steps-1):
        start_st = time.time()
        
        ## CONTROL DECISION
        q_ref_func   = interp1d(np.arange(2, step = 1), np.tile(q_star.reshape(-1,1), 2), kind='nearest', fill_value='extrapolate', bounds_error=False)
      
        q            = rom.encode(plant.y_full).copy()
        ut, Jt       = nmpc.get_action(plant.t, q, q_ref_func)  
        
        # One-step plant integration and state measurement
        plant.advance(ut)
        plant.measure_state(std_noise)

        
        # Calculate the reward
        du            = ut - utm1
        rew_s         = -((plant.x - x_star).T @ nmpc.Qx @ (plant.x - x_star)).copy()
        rew_u         = -ut.T @ nmpc.Ru @ ut 
        rew_du        = -du.T @ nmpc.Rdu @ du
        rew           = rew_s + rew_u +rew_du
        
        # Store episodes results
        xh[_episode,:,_step+1]   = plant.x
        yh[_episode,:,_step+1]   = plant.y_sparse
        rewh[_episode,_step+1]   = rew
        uh[_episode,:,_step]     = ut
        J_mpch[_episode,_step]   = Jt
        end_st                   = time.time()
        el_time_st               = end_st - start_st
        utm1                     = ut.copy()

        # Show progress during control application 
        print(f"ep: {_episode} - t = {_step*plant.dt:>5.2f} - {'rew_s':<5}= {rew_s:>7.2f} - {'rew_u':<5}= {rew_u:>7.2f} - {'rew_du':<5}= {rew_du:>7.2f} - J_nmpc = {Jt:>7.1f}, ElTime = {el_time_st:>7.3f}")
        
        if _step % 2 == 0:
            hf    = plt.figure(2,figsize=(6, 5))
            plt.clf()
            ax3d  = hf.add_subplot(111, projection='3d')
            downs = 1
            X     = plant_model.csi_x_grid[::downs, ::downs]
            Y     = plant_model.csi_y_grid[::downs, ::downs]
            Z     = plant.x.reshape(plant_model.csi_x_grid.shape)[::downs, ::downs]
            Z2    = rom.decode(q_ref_func(0)).reshape(plant_model.csi_x_grid.shape)[::downs, ::downs]  
            surf  = ax3d.plot_surface(X, Y, Z, cmap=flat_red, edgecolor='k', linewidth=0.2, alpha=0.5, label = 'Ground truth')
            surf2 = ax3d.plot_surface(X, Y, Z2, cmap=flat_blue, edgecolor='k', linewidth=0.2, alpha=0.5, label = 'target')
            ax3d.set_title(rf'ep: {_episode} - mpc control: states - $t = {_step * dt:.3f}$')
            ax3d.set_xlabel(r"$\xi$")
            ax3d.set_ylabel(r"$\eta$")
            ax3d.set_zlabel(r"$x(\xi, \eta, t)$")
            ax3d.legend(loc='lower left')
            plt.tight_layout()
            plt.draw()
            plt.show()
            plt.pause(0.1) 
            
            
            plt.figure(3, figsize=(8, 5))
            plt.clf()
            plt.plot(th[:_step], uh[_episode,:,:_step].T, label='true state', linewidth=1)
            plt.ylim(-15, 15)
            plt.xlabel(r"$t$")
            plt.ylabel(r"$u(t)$")
            plt.title(rf'ep: {_episode} - mpc control: actions - $t = {_step * dt:.3f}$')
            plt.draw()
            plt.show()
            plt.pause(0.1)
            
        
    gc.collect()        
    end_ep              = time.time()
    el_time_ep          = end_ep - start_ep
    
    # Save the data 
    if save_data:
        filename = f"..//data//_KS2D//dataset_{dataset_number}//results//control//full_obs//control_results.h5"
        with h5py.File(filename, "w") as f:
            f.create_dataset("xh", data=xh[:_episode+1,:,:], compression="gzip")
            f.create_dataset("uh", data=uh[:_episode+1,:,:], compression="gzip")
            f.create_dataset("th", data=th, compression="gzip")
            f.create_dataset("x_star", data=x_star, compression="gzip")
            f.create_dataset("J_mpch", data=J_mpch[:_episode+1,:], compression="gzip")
            
            
    
    print(f'--info:  Episode {_episode} ended. Elapsed time: {round(el_time_ep,2)} s')
print(f'--info:  Control simulation ended.')