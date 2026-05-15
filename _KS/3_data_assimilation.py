"""
===============================================================================
File:         3_data_assimilation.py
Author:       Luigi Marra
Email:        luigi.marra@uc3m.es

Description:  
    Execution of the Unscented Kalman filter using the OpInf model and sparse and noisy measurement data on a validation dataset
    Implementation for the one dimensional Kuramoto-Sivashinsky equation

===============================================================================
"""



#%% Packages 
# --------------------------------------------- #
import os, h5py, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import numpy as np
import matplotlib.pyplot as plt
import general_utils.plant as plant_class
import general_utils.plant_model as plant_model_class
import general_utils.ukf_class as UKF
# --------------------------------------------- #



#%% Read a validation dataset
with h5py.File('..//data//_KS//validation_dataset.h5', 'r') as f:
    uh_val    = f['uh_val'][:]
    xh_val    = f['xh_val'][:]
    t_val     = f['t_val'][:] 


#%% Define parameters of the test
plant_type       = "KS" 
dt               = 0.1

ny               = 4                                                           # Number of sensors for the feedback
std_noise        = 0.1                                                         # Standard deviation of the measurement noise
sample_frequency = int(0.1/dt)                                                 # Correction interval of UKF
ind_start        = 0                                                           # Start index in the validation dataset
ind_end          = 5000                                                        # End index in the validation dataset
save_data        = False                                                       # Bool variable to save UKF estimation results


#%% Initialize plant model
model_path = '..//data//_KS//'
plant_model      = plant_model_class.plant(plant_type = plant_type, dt = dt, d_sparse = ny, model_path = model_path)        # Plant model 
plant            = plant_class.plant(plant_type = plant_type, dt = plant_model.dt, d_sparse = ny)  # Plant
rom              = plant_model.rom                                                                 # OpInf Reduced order model 
basis            = rom.basis.entries                                                               # POD basis for compression 
latent_dim       = rom.basis.reduced_state_dimension                                               # Dimension of the latent space
ukf              = UKF.UKFStateEstimator(plant_model, ny, std_noise)                               # UKF class object


#%% Initialize variables
n_step           = ind_end - ind_start
xh               = np.full([plant.n, n_step], np.nan)
yh               = np.full([ny, n_step], np.nan)
qh               = np.full([latent_dim, n_step], np.nan)

xhath            = np.full([plant.n, n_step], np.nan)
xstdh            = np.full([plant.n, n_step], np.nan)
qhath            = np.full([latent_dim, n_step], np.nan)
qstdh            = np.full([latent_dim, n_step], np.nan)

yresh            = np.full([ny, n_step], np.nan)
yrescovh         = np.full([ny, ny, n_step], np.nan)
y_idealh         = np.full([ny, n_step], np.nan)


x                = xh_val[:,ind_start]
y_ideal          = np.interp(plant_model.csi_y_sparse, plant_model.csi, x)
y                = y_ideal + np.random.normal(0, std_noise, size=plant.csi_y_sparse.shape)
q                = rom.encode(x)


#%%  Initialize the filter state with Gaussian Process Regression

ukf.initialize_x_gpr(y)


#%% Start recursive filtering ...

ind               = 0
xh[:,ind]         = x
yh[:,ind]         = y 
qh[:,ind]         = q
th                = t_val
xhath[:,ind]      = ukf.xhat0
xstdh[:,ind]      = np.diag(ukf.x0hat_cov_gpr)
qhath[:,ind]      = ukf.ukf.x
qstdh[:,ind]      = np.diag(ukf.ukf.P)
uh                = uh_val[:,ind_start:ind_end]
y_idealh[:,ind]   = y_ideal


for i in range(ind_start, ind_end-1):
    
    u           = uh_val[:,i]
    x           = xh_val[:,i+1]
    q           = rom.encode(x)
    y_ideal     = np.interp(plant_model.csi_y_sparse, plant_model.csi, x) 
    y           = y_ideal + np.random.normal(0, std_noise, size=plant.csi_y_sparse.shape)
    
    # Estimation (predict-correct) step
    bool_update = ((i + 1) % sample_frequency == 0)
    qhat, qcov, xhat, xcov, y_residual, residual_covar = ukf.predict_correct_step(u, y, bool_update) 

    
    # Store data
    ind += 1
    if bool_update:
        yh[:,ind]         = y
    xh[:,ind]         = x
    qh[:,ind]         = q

    xhath[:,ind]      = xhat
    xstdh[:,ind]      = np.diag(xcov)
    qhath[:,ind]      = qhat
    qstdh[:,ind]      = np.diag(qcov)
    yrescovh[:,:,ind] = residual_covar 
    yresh[:,ind]      = y_residual
    y_idealh[:,ind]   = y_ideal
    
    # Plot online results
    if i % 50 == 0:
        plt.figure(1)
        plt.clf()
        plt.plot(plant.csi, x, 'r--', label = 'Ground thruth')
        plt.plot(plant.csi, xhat,'b-', label = 'UKF estim. mean')
        plt.fill_between(plant.csi, xhat - np.sqrt(np.diag(xcov)), xhat + np.sqrt(np.diag(xcov)), color='blue', alpha=0.2, label = 'ukf estim. uncertainty (1 $\sigma$)')
        plt.plot(plant.csi_y_sparse, y ,'k.', markersize = 15, label = 'Measurement')
        plt.legend(loc='lower left')
        plt.title(f't = {i*dt}')
        plt.xlabel(r"$\xi$")
        plt.xlabel(r"$x(\xi,t)$")
        plt.xlim([0, 22])
        plt.ylim([-6, 6])
        plt.show()
        plt.pause(0.05)
        
      

#%% Save the data 
if save_data:
    filename         = f'..//data//_KS//results//ukf//ukf_sens{int(ny)}_noise{int(std_noise*10)}_samplfreq{int(sample_frequency)}.h5'
    with h5py.File(filename, "w") as f:
        f.create_dataset("sample_frequency", data=sample_frequency)
        f.create_dataset("std_noise",        data=std_noise)
        f.create_dataset("ny",               data=ny)
        
        f.create_dataset("xh", data=xh, compression="gzip")
        f.create_dataset("uh", data=uh, compression="gzip")
        f.create_dataset("th", data=th, compression="gzip")
        f.create_dataset("yh", data=yh, compression="gzip")
        f.create_dataset("qh", data=qh, compression="gzip")
        
        f.create_dataset("xhath", data=xhath, compression="gzip")
        f.create_dataset("xstdh", data=xstdh, compression="gzip")
        f.create_dataset("qhath", data=qhath, compression="gzip")
        f.create_dataset("qstdh", data=qstdh, compression="gzip")
        
        f.create_dataset("yresh", data=yresh, compression="gzip")
        f.create_dataset("yrescovh", data=yrescovh, compression="gzip")
        f.create_dataset("y_idealh", data=y_idealh, compression="gzip")
        f.create_dataset("csi_y", data=plant_model.csi_y_sparse, compression="gzip")
        f.create_dataset("csi", data=plant_model.csi, compression="gzip")


        