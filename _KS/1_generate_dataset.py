"""
===============================================================================
File:         1_generate_dataset.py
Author:       Luigi Marra
Email:        luigi.marra@uc3m.es

Description:  
    Generates the dataset to be used for data-driven modeling and estimation.
    The dataset is split into training and validation sets.
    This code refers to the one dimensional Kuramoto-Sivashinsky equation

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
from general_utils.utils import gen_IC, fourier_filtered_input, psd
#------


#%% Initialize the 1D KS model
dt                = 0.1                                                        # Timestep of the model
action_std        = 3                                                          # standard deviation of the actions for dataset generation
plant             = plant_class.plant(plant_type = 'KS', dt = dt)              # Plant class of the system
co_freq           = 2                                                          # Cut-off frequency for action smoothing
save_bool         = False                                                      # Bool variable to decide if saving training/validation datasets



#%% Integrate the model to generate the training dataset
T_tr              = 1000                                                       # Length of the training dataset in c.u.
nt_tr             = int(T_tr/dt)                                               # Number of snapshots of the training dataset
t_tr              = np.arange(0, dt*nt_tr, step = dt)                          # Time domain of the training dataset


# Wait some time to get a physically meaningful IC (reach the system attractor)
x0                = gen_IC(plant, amplitude = 6, cutoff_freq = 0.5)            
t_w               = 100
xx,_,_,_          = plant.gen_dataset(x0 = x0, Tend = t_w, actions = np.full((plant.m, int(t_w/dt)), 0))
x0_tr             = xx[:,-1]                                                   # Initial condition used for the dataset


# Define actuation
uh_tr      = np.full((plant.m, nt_tr), np.nan)
N_ramp     = 10
i_no_act   = int(200/dt)                                                       # Length of the unactuated part of the dataset
for i in range(plant.m):
    action                = fourier_filtered_input(t_tr[i_no_act:], cutoff_freq = co_freq) 
    norm_action           = (action - np.mean(action, keepdims=True))/(np.std(action, keepdims=True) + 1e-8)
    norm_action[:N_ramp] *= np.arange(N_ramp)/(N_ramp-1)
    uh_tr[i,:]            = np.concatenate((np.zeros(i_no_act), norm_action*action_std))  # Matrix with time series of control actuations


# Generate the dataset for training
xh_tr, _, _, _  = plant.gen_dataset(x0 = x0_tr, Tend = T_tr, actions = uh_tr)  # Matrix with time series of system full states
fcoeff_tr       = psd(xh_tr, 0, indices = [1,2,3])                             #Coefficients of the states projected in the main three Fourier modes



#%% Integrate the model to generate the validation dataset (same procedure of the training dataset...)
T_val             = T_tr
nt_val            = int(T_val/dt)
x0_val            = xx[:,-500]
t_val             = np.arange(0, dt*nt_val, step = dt)
uh_val            = np.full((plant.m, nt_val), np.nan)


# Define actuation
for i in range(plant.m):
    action                = fourier_filtered_input(t_val[i_no_act:], cutoff_freq = co_freq)
    norm_action           = (action - np.mean(action, keepdims=True))/(np.std(action, keepdims=True) + 1e-8)
    norm_action[:N_ramp] *= np.arange(N_ramp)/(N_ramp-1)
    uh_val[i,:]           = np.concatenate((np.zeros(i_no_act), norm_action*action_std))


# Generate the dataset for training
xh_val, _, _, _  = plant.gen_dataset(x0 = x0_val, Tend = T_tr, actions = uh_val)
fcoeff_val       = psd(xh_val, 0, indices = [1,2,3])



#%% Save the data...

if save_bool:
    # Save the training dataset ...
    with h5py.File('data//_KS//training_dataset.h5', 'w') as f:
        f.create_dataset('uh_tr',      data=uh_tr);
        f.create_dataset('xh_tr',      data=xh_tr);
        f.create_dataset('t_tr',       data=t_tr); 
        f.create_dataset('fcoeff_tr',  data=fcoeff_tr); 
        print('Saved trainin dataset ...')
        
    # Save the validation dataset ...
    with h5py.File('data//_KS//validation_dataset.h5', 'w') as f:
        f.create_dataset('uh_val',      data=uh_val);
        f.create_dataset('xh_val',      data=xh_val);
        f.create_dataset('t_val',       data=t_val); 
        f.create_dataset('fcoeff_val',  data=fcoeff_val);
        print('Saved validation dataset ...')

