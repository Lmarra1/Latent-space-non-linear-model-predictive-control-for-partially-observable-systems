"""
===============================================================================
File:         1_generate_dataset.py
Author:       Luigi Marra
Email:        luigi.marra@uc3m.es

Description:  
    Generates the dataset to be used for data-driven modeling and estimation.
    The dataset is split into training and validation sets.
    This code refers to the two dimensional Kuramoto-Sivashinsky equation

===============================================================================
"""
#%% Packages 
#------
import numpy as np
import h5py, sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
#------
import general_utils.plant as plant_class
from general_utils.utils import fourier_filtered_input
#------



#%% Initialize the 2DKS model
dt                = 0.02                                                       # Timestep of the model
action_std        = 3                                                          # standard deviation of the actions for dataset generation
plant             = plant_class.plant(plant_type = 'KS2D', dt = dt)            # Plant class of the system
co_freq           = 1                                                          # Cut-off frequency for action smoothing
save_bool         = False                                                      # Bool variable to decide if saving training/validation datasets
rm_spat_mean_act  = True                                                       # Bool to decide if removing the spatial mean of the actuation
dataset_number    = 1                                                          # Number of the training dataset



#%% Check existing saving directory...
save_dir = f'..//data//_KS2D//dataset_{dataset_number}'
os.makedirs(save_dir, exist_ok=True)



#%% Integrate the model to generate the training dataset
T_tr              = 3000                                                       # Length of the training dataset in c.u.
nt_tr             = int(T_tr/dt)                                               # Number of snapshots of the training dataset
t_tr              = np.arange(0, dt*nt_tr, step = dt)                          # Time domain of the training dataset
dcsi_x            = plant.csi_x[1] - plant.csi_x[0]                            # Grid size of the integration domain 
dcsi_y            = plant.csi_y[1] - plant.csi_y[0]


# Wait some time to get a physically meaningful IC
x0                = np.sin(plant.csi_x_grid + plant.csi_y_grid) + np.sin(plant.csi_x_grid) + np.sin(plant.csi_y_grid)
t_w               = 500 
xx,_,_,_          = plant.gen_dataset(x0 = x0, Tend = t_w, actions = np.full((plant.m, int(t_w/dt)), 0))
x0_tr             = xx[:,-1]                                                   # Initial condition used for the dataset


# Define actuation
uh_tr             = np.full((plant.m, nt_tr), np.nan)
N_ramp            = 10
i_no_act          = int(500/dt)
for i in range(plant.m):
    action                = fourier_filtered_input(t_tr[i_no_act:], cutoff_freq = co_freq) 
    norm_action           = (action - np.mean(action, keepdims=True))/(np.std(action, keepdims=True) + 1e-8)
    norm_action[:N_ramp] *= np.arange(N_ramp)/(N_ramp-1)
    uh_tr[i,:]            = np.concatenate((np.zeros(i_no_act), norm_action*action_std))


# Remove spatial mean actuation
if rm_spat_mean_act:
    utrmean = np.mean(uh_tr,axis = 0)
    uh_tr = uh_tr - utrmean


# Generate the dataset for training
xh_tr, _, _, _  = plant.gen_dataset(x0 = x0_tr, Tend = T_tr, actions = uh_tr)
en_tr           = np.sum(xh_tr**2, axis=0) * dcsi_x * dcsi_y


#%% Integrate the model to generate the validation dataset (same procedure of the training dataset...)
T_val             = T_tr
nt_val            = int(T_val/dt)
t_val             = np.arange(0, dt*nt_val, step = dt)
nt_val            = int(T_val/dt)


# Wait some time to get a physically meaningful IC
xx,_,_,_          = plant.gen_dataset(x0 = x0, Tend = t_w, actions = np.full((plant.m, int(t_w/dt)), 0))
x0_val            = xx[:,-1]


# Define actuation
uh_val            = np.full((plant.m, nt_val), np.nan)
for i in range(plant.m):
    action                = fourier_filtered_input(t_val[i_no_act:], cutoff_freq = co_freq)
    norm_action           = (action - np.mean(action, keepdims=True))/(np.std(action, keepdims=True) + 1e-8)
    norm_action[:N_ramp] *= np.arange(N_ramp)/(N_ramp-1)
    uh_val[i,:]           = np.concatenate((np.zeros(i_no_act), norm_action*action_std))


if rm_spat_mean_act:
    uvalmean = np.mean(uh_val,axis = 0)
    uh_val   = uh_val - uvalmean


# Generate the dataset for training
xh_val, _, _, _  = plant.gen_dataset(x0 = x0_val, Tend = T_val, actions = uh_val)
en_val           = np.sum(xh_val**2, axis=0) * dcsi_x * dcsi_y



#%% Save the data...

if save_bool:    
    
    # Save training dataset...
    with h5py.File(os.path.join(save_dir, 'training_dataset.h5'), 'w') as f:
        f.create_dataset('uh_tr',      data=uh_tr);
        f.create_dataset('xh_tr',      data=xh_tr);
        f.create_dataset('en_tr',      data=en_tr);
        f.create_dataset('t_tr',       data=t_tr);     
        print('Saved trainin dataset ...')
        
        
    # Save validation dataset...
    with h5py.File(os.path.join(save_dir, 'validation_dataset.h5'), 'w') as f:
        f.create_dataset('uh_val',      data=uh_val);
        f.create_dataset('xh_val',      data=xh_val);
        f.create_dataset('t_val',       data=t_val);
        f.create_dataset('en_val',      data=en_val);
        print('Saved validation dataset ...')
        
        
    # Save simulation parameters...
    with h5py.File(os.path.join(save_dir, 'params.h5'), 'w') as f:
        f.create_dataset('L', data=plant.L)
        f.create_dataset('ni_1', data=plant.ni_1)
        f.create_dataset('ni_2', data=plant.ni_2)
        f.create_dataset('n_act_grid', data=plant.n_act_grid)
        f.create_dataset('act_width', data=plant.act_width)
        f.create_dataset('n_csi_x', data=plant.n_csi_x)
        f.create_dataset('n_csi_y', data=plant.n_csi_y)
        f.create_dataset('dti', data=plant.dti)
        f.create_dataset('dtrom', data=dt)
        print('Saved simulation parameters...')
        
        
    # Save parameters relative to the grid...
    with h5py.File(os.path.join(save_dir, 'grids.h5'), "w") as f:
        f.create_dataset("csi_x", data=plant.csi_x, compression="gzip")
        f.create_dataset("csi_y", data=plant.csi_y, compression="gzip")
        f.create_dataset("csi_x_grid", data=plant.csi_x_grid, compression="gzip")
        f.create_dataset("csi_y_grid", data=plant.csi_y_grid, compression="gzip")
        f.create_dataset("control_basis_grid", data=plant.control_basis_grid, compression="gzip")
        f.create_dataset("control_basis", data=plant.control_basis, compression="gzip")
        f.create_dataset("act_centers", data=plant.ks2d_integrator.act_centers, compression="gzip")
        f.create_dataset("act_width", data=plant.act_width)
        f.create_dataset("csi_x_sens", data=plant.csi_x_sens)
        f.create_dataset("csi_y_sens", data=plant.csi_y_sens)
        print('Saved grid parameters...')
        
    
    
    






