"""
===============================================================================
File:         4_nmpc_latent_full_observable.py
Author:       Luigi Marra
Email:        luigi.marra@uc3m.es

Description:  
    Implementation of Model Predictive Control in full observability in the latent space from OpInf
    Implementation for the one dimensional Kuramoto-Sivashinsky equation

===============================================================================
"""



#%% Packages 
import time, gc, os, h5py, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import general_utils.plant as plant_class
import general_utils.plant_model as plant_model_class
import general_utils.nmpc_latent_casadi as nmpc_class

# --------------------------------------------- #
# Util functions
from general_utils.utils import psd
# --------------------------------------------- #



#%% Parameters of the plant 
plant_type       = "KS" 
dt               = 0.1       # Time interval between each action update (control timestep)
std_noise        = 0.0


# Initialize the plant model for mpc and also save the reduced order model
model_path       = '..//data//_KS//'
plant_model      = plant_model_class.plant(plant_type = plant_type, dt = dt, model_path = model_path)
plant            = plant_class.plant(plant_type = plant_type, dt = plant_model.dt)
rom              = plant_model.rom
latent_dim       = rom.basis.reduced_state_dimension


# Read the control objective and compressed counterparts
with h5py.File('..//data//_KS//invariant_solutions.h5', 'r') as f:
    E0    = f['E0'][:]
    E1    = f['E1'][:]
    E2    = f['E2'][:]
    E3    = f['E3'][:]
    
q0 = rom.encode(E0)
q1 = rom.encode(E1)
q2 = rom.encode(E2)
q3 = rom.encode(E3)


eq_target = 3    # 0: E_0;    1: E_1;    2: E_2;    3: E_3;    other: time dependent trajectory
print(f"target {eq_target}")
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
n_episodes       = 250                                                         # Number of control trajectories
duration_episode = 50                                                          # Lenght of each episode
n_steps          = int(duration_episode/plant.dt)
save_data        = False                                                       # Bool variable to save control results



#%%  Variable to store data during the control
xh               = np.full([n_episodes, plant.n,        n_steps], np.nan)
yh               = np.full([n_episodes, plant.d_sparse, n_steps], np.nan)
uh               = np.full([n_episodes, plant.m,        n_steps], np.nan)
qh               = np.full([n_episodes, latent_dim,     n_steps], np.nan)
coeffsh          = np.full([n_episodes, 3,              n_steps], np.nan)
th               = np.arange(0, duration_episode, step = plant.dt)
J_mpch           = np.full([n_episodes, n_steps], np.nan)
utm1             = np.zeros(plant_model.m)
traj_ref_h       = rom.decode(q_ref_func(th))
coeffs_traj      = psd(traj_ref_h, axis = 0)



#%%  Read initial conditions from files
with h5py.File('..//data//_KS//initial_conditions.h5', 'r') as f:
    ICs    = f['IC'][:]

#%%  Start simulation...

print('--info:  Control simulation started...')

for _episode in range(n_episodes):
    
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
    
    # Store initial data
    xh[_episode,:,0]          = plant.x
    yh[_episode,:,0]          = plant.y_sparse
    qh[_episode,:,0]          = q0
    coeffsh[_episode,:,0]     = psd(plant.x, 0)

    
    # Start the episode
    for _step in range(n_steps-1):
        start_st = time.time()
    
        q            = rom.encode(plant.y_full).copy()
        ut, Jt       = nmpc.get_action(plant.t, q, q_ref_func) 
        
        # One-step plant integration and state measurement
        plant.advance(ut)
        plant.measure_state(std_noise)
        
        q_star = q_ref_func(plant.t)
        x_star = rom.decode(q_star)
        
        # Store episodes results
        xh[_episode,:,_step+1]       = plant.x
        yh[_episode,:,_step+1]       = plant.y_sparse
        qh[_episode,:,_step+1]       = rom.encode(plant.x)
        uh[_episode,:,_step]         = ut
        coeffsh[_episode,:,_step+1]  = psd(plant.x, 0)
        J_mpch[_episode,_step]       = Jt
        end_st                       = time.time()
        el_time_st                   = end_st - start_st
        utm1                         = ut.copy()

        # Show progress during control application 
        print(f"ep: {_episode} - t = {_step*plant.dt:>5.3f} - J_nmpc = {Jt:>7.3f}, ElTime = {el_time_st:>7.3f}")
        
        if _step % 10 == 0 and _step > 0:
            hf = plt.figure(2, figsize=(8, 5))
            plt.clf()
            plt.plot(plant.csi, plant.x, 'k-', label='State', linewidth=1)
            plt.plot(plant.csi, x_star, 'r-', label='Target', linewidth=1)
            plt.legend(ncol=2, frameon=False, loc='lower left')
            plt.ylim(-7, 6)
            plt.xlim(0, 22)
            plt.xlabel(r"$\xi$")
            plt.ylabel(r"$x(\xi, t)$")
            plt.title(rf'ep: {_episode} - mpc control: states - $t = {_step * dt:.1f}$')
            plt.show()
            plt.pause(0.01)
            
            
            hf = plt.figure(3, figsize=(8, 5))
            plt.clf()
            plt.plot(uh[_episode,:,:].T, label='true state', linewidth=1)
            plt.ylim(-10, 10)
            plt.xlabel(r"$t$")
            plt.ylabel(r"$u$")
            plt.title(rf'ep: {_episode} - mpc control: actions - $t = {_step * dt:.1f}$')
            plt.show()
            plt.pause(0.01)
            
            
            hf = plt.figure(4, figsize=(7, 5))
            plt.clf()
            ax = hf.add_subplot(111, projection='3d')
            ax.plot(coeffsh[_episode, 0, :], coeffsh[_episode, 1, :], coeffsh[_episode, 2, :], 'b-', linewidth = 2)
            ax.plot(coeffs_traj[0, :], coeffs_traj[1, :], coeffs_traj[2, :], 'r.')
            ax.set_xlabel(r"$e_1$")
            ax.set_ylabel(r"$e_2$")
            ax.set_zlabel(r"$e_3$")
            ax.set_title(rf"Trajectory in Fourier space - ep: {_episode} - $t = {_step * dt:.1f}$")
            ax.view_init(elev=30, azim=135)
            plt.tight_layout()
            plt.show()
            plt.pause(0.01)
            
            
            
        
    gc.collect()        
    end_ep              = time.time()
    el_time_ep          = end_ep - start_ep
    
    print(f'--info:  Episode {_episode} ended. Elapsed time: {round(el_time_ep,2)} s')
print(f'--info:  Control simulation ended.')

# Save the data 
if save_data:
    filename = f"..//data//_KS//control_full_obs_E{eq_target}.h5"
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

