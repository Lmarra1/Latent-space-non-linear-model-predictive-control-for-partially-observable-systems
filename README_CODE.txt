# Latent-Space Model Predictive Control for Partially-Observable Systems

## Overview

This repository contains Python code for the implementation of the nonlinear Model Predictive Control (MPC) architecture for partially-observable systems. The code learns and implements a surrogate model in a data-driven manner via Operator Inference (OpInf). It provides a user-defined MPC implementation in CasADi with a state estimator based on the Unscented Kalman Filter (UKF). The controller is tested on both one-dimensional and two-dimensional Kuramoto-Sivashinsky systems.



## Key Features

- Latent-space MPC implemented using the CasADi symbolic framework
- Operator Inference (OpInf) for reduced-order model identification
- UKF-based state estimation from sparse and noisy measurements
- Ready-to-run examples for 1D and 2D KS systems


## Contact Information

- Author: Luigi Marra
- Email: lmarra@pa.uc3m.es


## Repository Information

- GitHub epository URL: TBD


## Associated Research Paper

- Title: Latent-Space Model Predictive Control for Partially-Observable Systems
- Authors: Luigi Marra, Onofrio Semeraro, Lionel Mathelin, Andrea Meilán-Vila, Stefano Discetti
- Journal: under consideration for submission in the Journal "Proceedings of the Royal Society A"
- Year: TBD
- DOI: TBD


## Requirements

The packages installed in this environment are:

It is recommended to create a virtual environment before installing the following dependencies to run the code, mainly written in Python. 

The dependencies to be installed are the following:
1) numpy
2) matplotlib
3) dill
4) scipy
5) h5py
6) Shenfun (version 4.2.2) 
from the link: https://shenfun.readthedocs.io/en/latest/installation.html# 
or:            https://pypi.org/project/shenfun/#history
7) casadi (version 3.7.2) 
from the link: https://web.casadi.org/get/ 
or:            https://pypi.org/project/casadi/#history
8) filterpy (version 1.4.5) 
from the link: https://filterpy.readthedocs.io/en/latest/ 
or:            https://pypi.org/project/filterpy/#history
9) opinf (version 0.5.16) 
from the link: https://willcox-research-group.github.io/rom-operator-inference-Python3/source/opinf/installation.html 
or:            https://pypi.org/project/opinf/#history


If you are using a Linux system and want to create a Python virtual environment using venv, you can follow the steps below. This will create an isolated environment and install all the required packages:

'''
# Create virtual environment
python3 -m venv myenv_mpc

# Activate it
source myenv_mpc/bin/activate

# Upgrade pip, setuptools, and wheel
pip install --upgrade pip setuptools wheel

# Install required packages
pip install numpy \
            matplotlib \
            dill \
            scipy \
            h5py \
            sympy \
	    numba

# Shenfun (version 4.2.2)
pip install shenfun==4.2.2

# CasADi (version 3.7.2)
pip install casadi==3.7.2

# FilterPy (version 1.4.5)
pip install filterpy==1.4.5

# OpInf (version 0.5.16)
pip install opinf==0.5.16

'''


## Codes Overview

The repository contains the following folders:

1) _KS: Contains the code to run the control algorithm on the one-dimensional Kuramoto-Sivashinsky equation.

2) _KS2D: Contains the code to run the control algorithm on the two-dimensional Kuramoto-Sivashinsky equation.

3) general_utils: Contains utility functions to run the control methods on the proposed test cases.

4) data: Contains the data required to run the control on the test cases without the need to generate the dataset for system model identification. The model used by MPC is already provided.

Folders 1) and 2) contain the following main scripts:
- 1_generate_dataset.py
  Creates the training and validation dataset to later learn the reduced-order model with Operator Inference.
- 2_operator_inference.py
  Performs POD to identify the basis where the model is learned with Operator Inference.
- 3_data_assimilation.py
  Performs data assimilation using UKF to estimate the full and latent state from sparse measurements over a validation dataset.
- 4_nmpc_latent_full_observable.py
  Implements MPC in the OpInf (POD) latent coordinates using the full state as feedback.
- 5_nmpc_latent_UKF.py
  Implements MPC in the OpInf (POD) latent coordinates using sparse measurements as feedback and UKF as the estimation layer.

Folder 3) contains the following utility functions:
- nmpc_latent_casadi.py
  Class implementing the nonlinear MPC code using CasADi symbolic problem definition.
- plant.py
  General class defining all the plants used in this work.
- plant_model.py
  General class containing plant models used in this work.
- ukf_class.py
  FilterPy class to run the UKF for the one-dimensional case.
- ukf_class_2D.py
  FilterPy class to run the UKF for the two-dimensional case.
- KS2D_general.py
  Class integrating the solution of the two-dimensional Kuramoto-Sivashinsky equation using the spectral Galerkin method implemented in the Shenfun library.




You can run the control in full or partial observability using one of the following scripts:
_KS/4_nmpc_latent_full_observable.py  
_KS/5_nmpc_latent_UKF.py  

or ...

_KS2D/4_nmpc_latent_full_observable.py  
_KS2D/5_nmpc_latent_UKF.py  


ENJOY!!!



## Funding 

The authors acknowledge the support  from the funding under ‘Orden 3789/2022, del Vicepresidente, Consejero de Educación y Universidades, por la que se convocan ayudas para la contratación de personal investigador predoctoral en formación para el año 2022’.



## Issues and Feedback

If you encounter any issues or have feedback regarding this code, please open an issue on our GitHub repository or send an email to the authors. Your insights and suggestions are valuable and appreciated.












