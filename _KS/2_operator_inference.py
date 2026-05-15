"""
===============================================================================
File:         2_operator_inference.py
Author:       Luigi Marra
Email:        luigi.marra@uc3m.es

Description:  
    Uses the dataset to create a reduced order model with Operator Inference.
    Model for the one dimensional Kuramoto-Sivashinsky equation

===============================================================================
"""



#%% Packages 
#------
import numpy as np
import h5py, sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import matplotlib.pyplot as plt
import opinf
import dill
#------
import general_utils.plant as plant_class
#------


#%% Read training and validation dataset
with h5py.File('..//data//_KS//training_dataset.h5', 'r') as f:
    uh_tr      = f['uh_tr'][:]
    xh_tr      = f['xh_tr'][:]
    t_tr       = f['t_tr'][:]
    
with h5py.File('..//data//_KS//validation_dataset.h5', 'r') as f:
    uh_val    = f['uh_val'][:]
    xh_val    = f['xh_val'][:]
    t_val     = f['t_val'][:] 

dt                = 0.1
plant             = plant_class.plant(plant_type = 'KS', dt = dt)
csi               = plant.csi
nt_tr             = xh_tr.shape[1]
x_tr_mean         = np.mean(xh_tr, axis = 1)
xh_tr_centred     = xh_tr - x_tr_mean[:, np.newaxis]
nt_tr             = len(t_tr)
nt_val            = len(t_val)


#%% Perform POD

truncation_mode = 1 # 0: selecting the cumulated energy; other: selecting the number of modes

if truncation_mode == 0:
    cum_energy   = 0.9999
    basis        = opinf.basis.PODBasis(cumulative_energy=cum_energy)
else:
    num_vectors  = 14
    basis        = opinf.basis.PODBasis(num_vectors=num_vectors)


X  = xh_tr
basis.fit(X)
print('POD done...')
print(basis)

U            = basis.leftvecs
Sigma_scaled = basis.svdvals
V            = basis.rightvecs
Sigma        = np.diag(U.T@X@V)



#%% Save POD data
save_bool = False

if save_bool:
    with h5py.File('..//data//_KS//pod.h5', 'w') as f:
        f.create_dataset('U', data=U);
        f.create_dataset('Sigma', data=Sigma);
        f.create_dataset('Sigma_scaled', data=Sigma_scaled);
        f.create_dataset('V',  data=V); 
        f.create_dataset('nt',  data=nt_tr); 
        f.create_dataset('ns',  data=plant.n); 
        f.create_dataset('r',  data=basis.reduced_state_dimension);
        f.create_dataset('energy_retained',  data=basis.cumulative_energy);
        print('POD data saved...')



#%% Fit the parameters of OpInf model for different regularizers

#- "c": Constant    Operator
#- "A": Linear      Operator
#- "H": Quadratic   Operator
#- "G": Cubic       Operator
#- "B": Input       Operator
#- "N": State-Input Operator

operators = 'cAHGBN'

def create_rom(reg):
    rom = opinf.ROM(
        basis=basis,
        model=opinf.models.DiscreteModel(operators,
        solver=opinf.lstsq.L2Solver(regularizer=reg),
        ),
    )
    return rom

# Evaluate the model on the training dataset to determine teh regularization coefficient via sampling
n_pred        = 50                                                             # Number of trajectories for evaluation
nt_pred       = 1000                                                           # Length of each trajectory
index_ic      = np.linspace(0, nt_tr-nt_pred , num=n_pred, dtype=int)
n_reg         = 100
regularizers  = np.logspace(-5, 1.2, num = n_reg)                              # Regularizers sampled
xh_hat        = np.full((n_reg, n_pred, plant.n, nt_pred), np.nan)
xh_hat_ref    = np.full((n_pred, plant.n, nt_pred), np.nan)
t_pred        = np.arange(nt_pred)*dt 

for j in range(n_pred):
    xh_hat_ref[j,:,:]  = xh_tr[:,index_ic[j]:index_ic[j]+nt_pred]
    
rom = create_rom(reg=0)  
rom.fit(xh_tr, inputs=uh_tr)   
for i in range(n_reg):
    rom.model.solver.regularizer = regularizers[i]
    with opinf.utils.TimedBlock(f"Fitting OpInf ROM with regularizer {i+1}/{n_reg}"):
        rom.model.refit()
    for j in range(n_pred):
        xh_hat[i,j,:,:]  = rom.predict(xh_tr[:,index_ic[j]], niters = nt_pred, inputs=uh_tr[:,index_ic[j]:])
        
norm_ref    = np.linalg.norm(xh_hat_ref, axis = 1)
err         = np.mean(np.linalg.norm(xh_hat - xh_hat_ref[np.newaxis, :, :, :], axis = 2) / norm_ref[np.newaxis, :, :], axis = 1)

meanerr     = np.mean(err, axis = 1)
minerr      = np.nanmin(meanerr)
arg_min     = np.nanargmin(meanerr)
regularizer = regularizers[arg_min]


plt.figure()
plt.plot(regularizers, meanerr)
plt.plot(regularizer, meanerr[arg_min], 'r*')
plt.xlabel('Regularizers')
plt.ylabel('Error')
plt.show()



#%% Select the best regularizer for the L2 optimization
regularizer = 0.886
rom = create_rom(reg=regularizer)
with opinf.utils.TimedBlock(f"Fitting OpInf ROM with regularizer l = {regularizer}"):
    rom.fit(xh_tr , inputs=uh_tr)



#%% Save the model 
if save_bool:
    with open('..//data//_KS//rom.dill', 'wb') as f:
        dill.dump(rom, f)
    with open('..//data//_KS//basis.dill', 'wb') as f:
            dill.dump(basis, f)






