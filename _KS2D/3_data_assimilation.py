"""
===============================================================================
File:         3_data_assimilation.py
Author:       Luigi Marra
Email:        luigi.marra@uc3m.es

Description:  
    Execution of the Unscented Kalman filter using the OpInf model and sparse and noisy measurement data on a validation dataset
    Implementation for the two dimensional Kuramoto-Sivashinsky equation

===============================================================================
"""



#%% Packages 
#------
import numpy as np
import matplotlib.pyplot as plt
import h5py, dill, os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
#------
from scipy.interpolate import RegularGridInterpolator
from matplotlib import cm, colors
import general_utils.plant_model as plant_model_class
import general_utils.ukf_class_2D as UKF

#------
flat_red     = colors.ListedColormap(['red'])
flat_blue    = colors.ListedColormap(['blue'])



#%% Read the validation dataset and other parameters
dataset_number = 1
model_path = f'..//data//_KS2D//dataset_{dataset_number}'
with h5py.File(os.path.join(model_path, 'validation_dataset.h5'), 'r') as f:
    uh_val      = f['uh_val'][:]
    xh_val      = f['xh_val'][:]
    t_val       = f['t_val'][:]
    en_val      = f['en_val'][:]

# with h5py.File(os.path.join(model_path, 'energy_validation.h5'), 'r') as f:
#    en_val = f['en_val'][:]

with open(os.path.join(model_path, 'rom.dill'), 'rb') as f:
    rom = dill.load(f)
with h5py.File(os.path.join(model_path, 'params.h5'), 'r') as f:
    dt    = f['dtrom'][()];


#%%#%% Define parameters of the test

tag_name = 'forced' # If using a controlled or free segment of the training dataset

if tag_name == 'forced':
    ind_start        = int(500/dt)
    ind_end          = int(520/dt)
elif tag_name == 'unforced':
    ind_start        = int(450/dt)
    ind_end          = int(470/dt)

n_sens_grid      = 3                                                           # Number of sensors along one side of the grid: n_sens_grid x n_sens_grid
std_noise        = 0.5                                                         # Standard deviation of the measurement noise
sample_frequency = int(0.1/dt)                                                 # Correction interval of UKF

save_data        = False



#%% Initialize plant model and other parameters
nx              = xh_val.shape[0]
nu              = uh_val.shape[0]
plant_model     = plant_model_class.plant(plant_type = 'KS2D', dt = dt, model_path = model_path, n_sens_grid = n_sens_grid)
rom             = plant_model.rom
latent_dim      = rom.basis.reduced_state_dimension
csi_x           = plant_model.csi_x
csi_y           = plant_model.csi_y
csi_x_grid      = plant_model.csi_x_grid
csi_y_grid      = plant_model.csi_y_grid

dcsi_x          = csi_x[1] - csi_x[0]
dcsi_y          = csi_y[1] - csi_y[0]

csi_x_sens_grid = plant_model.csi_x_sens_grid
csi_y_sens_grid = plant_model.csi_y_sens_grid
ny              = np.prod(csi_y_sens_grid.shape)
ukf             = UKF.UKFStateEstimator(plant_model, ny, std_noise)


def measure(x):
    y  = RegularGridInterpolator((csi_x, csi_y), x.reshape(csi_x_grid.shape))(np.column_stack([csi_x_sens_grid.ravel(), csi_y_sens_grid.ravel()]))
    y += np.random.normal(0, std_noise, size=y.shape)
    return y.copy()

def measure_ideal(x):
    y  = RegularGridInterpolator((csi_x, csi_y), x.reshape(csi_x_grid.shape))(np.column_stack([csi_x_sens_grid.ravel(), csi_y_sens_grid.ravel()]))
    return y.copy()

def en_calc(x):
    return np.sum(x**2) * dcsi_x * dcsi_y



#%% Initialize variables
n_step           = ind_end - ind_start
xh               = np.full([xh_val.shape[0], n_step], np.nan)
yh               = np.full([ny, n_step], np.nan)
uh               = uh_val.copy()
qh               = np.full([latent_dim, n_step], np.nan)

xhath            = np.full([xh_val.shape[0], n_step], np.nan)
xstdh            = np.full([xh_val.shape[0], n_step], np.nan)
qhath            = np.full([latent_dim, n_step], np.nan)
qstdh            = np.full([latent_dim, n_step], np.nan)
enhath           = np.full(n_step, np.nan)
yresh            = np.full([ny, n_step], np.nan)
yrescovh         = np.full([ny, ny, n_step], np.nan)
y_idealh         = np.full([ny, n_step], np.nan)

x                = xh_val[:,ind_start]
y_ideal          = measure_ideal(x)
y                = measure(x)
q                = rom.encode(x)


#%%  Initialize the filter state with Gaussian Process Regression
ukf.initialize_x_gpr(y)
enhat            = en_calc(rom.decode(ukf.ukf.x))



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
y_idealh[:,ind]   = y_ideal
enhath[ind]       = enhat

for i in range(ind_start, ind_end-1):
    
    u = uh_val[:,i]
    x = xh_val[:,i+1]
    q = rom.encode(x)
    y_ideal = measure_ideal(x)
    y       = measure(x)
    
    # Estimation (predict-correct) step
    bool_update = ((i + 1) % sample_frequency == 0)
    qhat, qcov, xhat, xcov, y_residual, residual_covar = ukf.predict_correct_step(u, y, bool_update)
    enhat = en_calc(xhat)

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
    enhath[ind]       = enhat
    
    yrescovh[:,:,ind] = residual_covar 
    yresh[:,ind]      = y_residual
    y_idealh[:,ind]   = y_ideal
    
    
    # Plot online results
    if i % int(1/0.02) == 0:
        hf = plt.figure(2, figsize=(12, 5))
        plt.clf()
    
        # --- 3D SURFACES ---
        ax3d = hf.add_subplot(1, 2, 1, projection='3d')
        downs = 1
        X  = plant_model.csi_x_grid[::downs, ::downs]
        Y  = plant_model.csi_y_grid[::downs, ::downs]
        Z  = xhat.reshape(plant_model.csi_x_grid.shape)[::downs, ::downs]
        Z2 = x.reshape(plant_model.csi_x_grid.shape)[::downs, ::downs]
    
        surf  = ax3d.plot_surface(X, Y, Z,  cmap=flat_red,  edgecolor='k', linewidth=0.2, alpha=0.5, label="UKF estim. mean")
        surf2 = ax3d.plot_surface(X, Y, Z2, cmap=flat_blue, edgecolor='k', linewidth=0.2, alpha=0.5, label="Ground thruth")
    
        # Sensor points
        Y_vals = y.reshape(csi_x_sens_grid.shape)
        ax3d.scatter(csi_x_sens_grid, csi_y_sens_grid, Y_vals,
                     c='k', marker='o', s=60, label='Measurement')
    
        ax3d.set_xlabel(r"$\xi$")
        ax3d.set_ylabel(r"$\eta$")
        ax3d.set_zlabel(r"$x(\xi, \eta, t)$")
        ax3d.legend(loc='lower left')
    
        # --- ENERGY TIME HISTORY PLOT ---
        ax2 = hf.add_subplot(1, 2, 2)
        ax2.plot(t_val[ind_start:i], en_val[ind_start:i],         label="Ground truth",     color="black", linewidth=2)
        ax2.plot(th[:ind] + t_val[ind_start], enhath[:ind],       label="UKF estim.",       color="red", linewidth=2)
    
        ax2.set_xlabel("t")
        ax2.set_ylabel("E(t)")
        ax2.legend()
        ax2.grid(True)
    
        plt.tight_layout()
        plt.show()
        plt.pause(0.1)




#%%
# Save the data 
if save_data:               
    filename         = f'..//data//_KS2D//dataset_{dataset_number}//results//ukf//ukf_' + tag_name + f'_sens{int(ny)}_noise{int(std_noise*10)}_samplfreq{int(sample_frequency)}.h5'
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
        f.create_dataset("enhath", data=enhath, compression="gzip")
        
        f.create_dataset("yresh", data=yresh, compression="gzip")
        f.create_dataset("yrescovh", data=yrescovh, compression="gzip")
        f.create_dataset("y_idealh", data=y_idealh, compression="gzip")
        
        f.create_dataset("csi_x_sens_grid", data=csi_x_sens_grid, compression="gzip")
        f.create_dataset("csi_y_sens_grid", data=csi_y_sens_grid, compression="gzip")


