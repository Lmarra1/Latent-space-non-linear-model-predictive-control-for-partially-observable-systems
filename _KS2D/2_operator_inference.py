"""
===============================================================================
File:         2_operator_inference.py
Author:       Luigi Marra
Email:        luigi.marra@uc3m.es

Description:  
    Uses the dataset to create a reduced order model with Operator Inference.
    Model for the two dimensional Kuramoto-Sivashinsky equation

===============================================================================
"""



#%%
#------
import numpy as np
import h5py, sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import matplotlib.pyplot as plt
import opinf
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm, colors
import dill
#------
import general_utils.plant as plant_class
#------


#%% Read training and validation dataset
dataset_number     = 1
downs_factor_snap  = 1
number_snap        = int(3000/0.02)

#% Read training and validation dataset
with h5py.File(f'..//data//_KS2D//dataset_{dataset_number}//training_dataset.h5', 'r') as f:
    uh_tr      = f['uh_tr'][:]; uh_tr = uh_tr[:,:number_snap:downs_factor_snap];
    xh_tr      = f['xh_tr'][:]; xh_tr = xh_tr[:,:number_snap:downs_factor_snap]
    t_tr       = f['t_tr'][:];  t_tr  = t_tr[:number_snap:downs_factor_snap]
    en_tr      = f['en_tr'][:]; en_tr = en_tr[:number_snap:downs_factor_snap]
    
with h5py.File(f'..//data//_KS2D//dataset_{dataset_number}//validation_dataset.h5', 'r') as f:
    uh_val    = f['uh_val'][:];  uh_val = uh_val[:,:number_snap:downs_factor_snap]
    xh_val    = f['xh_val'][:];  xh_val = xh_val[:,:number_snap:downs_factor_snap]
    t_val     = f['t_val'][:];   t_val  = t_val[:number_snap:downs_factor_snap]
    en_val      = f['en_val'][:]; en_val = en_val[:number_snap:downs_factor_snap]
    
with h5py.File(f'..//data//_KS2D//dataset_{dataset_number}//params.h5', 'r') as f:
    dtrom    = f['dtrom'][()];
    
#%% CHECK FROM HERE TO READ THE CORRECT DATA PLEASE
dt                = dtrom * downs_factor_snap
plant             = plant_class.plant(plant_type = 'KS2D', dt = dt)
nt_tr             = xh_tr.shape[1]
nt_val            = len(t_val)



#%% Perform POD

readPODdata      = False                                                       # Read or perform POD on the training dataset
truncation_mode  = 1                                                           # 0: selecting the cumulated energy; other: selecting the number of modes

if truncation_mode == 0:
    cum_energy   = 0.9999
    basis        = opinf.basis.PODBasis(cumulative_energy=cum_energy)
else:
    num_vectors  = 60
    basis        = opinf.basis.PODBasis(num_vectors=num_vectors)


if readPODdata:
    with h5py.File(f'..//data//_KS2D//dataset_{dataset_number}//pod.h5', 'r') as f:
        U              = f['U'][:]
        Sigma          = f['Sigma'][:]
        Sigma_scaled   = f['Sigma_scaled'][:]
        V              = f['V'][:]
        nt_tr          = int(f['nt'][()])
        ns             = int(f['ns'][()])
        r              = int(f['r'][()])
        energy_retained = float(f['energy_retained'][()])
    
    with open(f'..//data//_KS2D//dataset_{dataset_number}//basis.dill', 'rb') as f:
        basis = dill.load(f)
        print(basis)
        
else:
    
    print('starting POD...')
    basis.fit(xh_tr)
    print('POD done...')
    print(basis)
    
    U            = basis.leftvecs
    Sigma_scaled = basis.svdvals
    V            = basis.rightvecs
    sigma_1      = U[:, 0].T @ xh_tr @ V[:, 0]
    Sigma        = sigma_1 * Sigma_scaled
    
    save_bool   = True
    num_vectors = basis.reduced_state_dimension

    if save_bool:
        with h5py.File(f'..//data//_KS2D//dataset_{dataset_number}//pod.h5', 'w') as f:
            f.create_dataset('U', data=U);
            f.create_dataset('Sigma', data=Sigma);
            f.create_dataset('Sigma_scaled', data=Sigma_scaled);
            f.create_dataset('V',  data=V); 
            f.create_dataset('nt',  data=nt_tr); 
            f.create_dataset('ns',  data=plant.n); 
            f.create_dataset('r',  data=basis.reduced_state_dimension);
            f.create_dataset('energy_retained',  data=basis.cumulative_energy);
    
        with open(f'..//data//_KS2D//dataset_{dataset_number}//basis.dill', 'wb') as f:
            dill.dump(basis, f)


#%% Fit manually the regularizer by minimizing the prediction error

operators = 'cAHB'

#- "c": Constant    Operator
#- "A": Linear      Operator
#- "H": Quadratic   Operator
#- "G": Cubic       Operator
#- "B": Input       Operator
#- "N": State-Input Operator

rom = opinf.ROM(
    basis=basis,
    model=opinf.models.DiscreteModel(operators,
    solver=opinf.lstsq.L2Solver(regularizer=0),
    ),
)

print('fitting model...')
rom.fit(xh_tr, inputs=uh_tr, fit_basis = False)


#%% Fit the parameters of OpInf model for different regularizers

window_size        = int(5/dt)   
time_domains = []
states       = []
states_enc   = []
inputs       = []


# # If you want to do it on the training dataset
for start in range(0, xh_tr.shape[1] - window_size, window_size):
    end_train = start + window_size

    t_win = t_tr[    start:end_train]
    x_win = xh_tr[:, start:end_train]
    u_win = uh_tr[:, start:end_train]  
    q_win = rom.encode(x_win)

    time_domains.append(t_win)
    states.append(x_win)
    inputs.append(u_win)
    states_enc.append(q_win)


#%%
regularizers  = np.logspace(-2, 2, num = 100)
error = []
batch_size = len(states)
for i in range(len(regularizers)):
    rom.model.solver.regularizer = regularizers[i]
    rom.model.refit()
    predictions = []
    for j in range(batch_size):
        qhat = rom.model.predict(states_enc[j][:,0], niters = states_enc[j].shape[1], inputs = inputs[j])
        predictions.append(qhat)
    ppred = np.array(predictions)
    ttrue = np.array(states_enc)
    err   = np.mean(np.sqrt(np.mean((ppred - ttrue), axis = 1)**2))/np.mean(np.sqrt(np.mean((ttrue), axis = 1)**2))
    error.append(err)
    print(f'regularizer {i}/{len(regularizers)} - {np.round(regularizers[i],2)} - err: {err}')
    
    
error = np.array(error)

plt.figure()
plt.plot(regularizers, error)
plt.xlabel('Regularizers')
plt.ylabel('Error')


#%% Set the best regularizer
rom.model.solver.regularizer = 0.1
rom.model.refit()



#%% Save the rom

with open(f'..//data//_KS2D//dataset_{dataset_number}//rom.dill', 'wb') as f:
    dill.dump(rom, f)
    
    

    
    
    
    
    
    
    
    
    
    