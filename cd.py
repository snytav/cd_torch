import torch
# import mpmath
import numpy as np

n_el=5                               # Number of finite elements
n_gauss=2                            # Number of Gauss points
Pe=1                                 # Péclet number

x_f = 1.0
x_i = 0.0
L = x_f - x_i  # Domain length
L_el = L / n_el  # Length of a finite element

polynomial_degree=1                     # Shape functions polynomial degree
n_np=polynomial_degree*n_el+1            # Number of nodal points
dof_el=polynomial_degree+1;             # Number of DOFs per element
dof=n_np                                # Total number of DOFs
h=L_el/polynomial_degree                # Spatial step
FE_type='Galerkin'                      # Type of FE (Galerkin or Upwind)
n_np=polynomial_degree*n_el+1           # Number of nodal points
steps_per_el = 10                           # number of steps per element
dx_e=L_el/10                                 # Spatial step-numerical interp.
x_e=np.linspace(0,L_el,steps_per_el+1)  # Space vector-numerical interp.
a=1                                          # Convection velocity
v=a*h/(2*Pe)                                 # Diffusivity coefficient

from gauss import Gauss_parameters,test_functions_Gauss_points
from gauss import shape_functions_gauss_points

csi, w = Gauss_parameters(n_gauss)
# Transformation of coordinates for the Gauss integration points
x_gauss = [L_el / 2 * (1 + csi[n]) for n in range(n_gauss)]

# Jacobian of the transformation
J = h / 2
x = np.arange(x_i, x_f + h, h)  # Space vector.

if FE_type == 'Galerkin':
    beta = 0
# elif FE_type == 'Upwind':
    # beta = mpmath.coth(Pe) - 1 / Pe

# Computation of shape and test functions (and derivatives) at Gauss points
N, dN = shape_functions_gauss_points(csi)
W, dW = test_functions_Gauss_points(csi, beta)


from plot_eometry import plot_geometry
if n_el <= 10:
    plot_geometry(x, h, n_np, L, n_el, n_gauss, L_el, x_gauss)

# Plot of shape and test functions

import matplotlib.pyplot as plt

# Normalized domain
csi_plot = np.arange(-1, 1.01, 0.01)

# Shape functions
from gauss import  f_N,f_W
N_plot = f_N(csi_plot)
from plot import plot_shape_functions,plot_test_functions
plot_shape_functions(csi_plot, N_plot)

# Test functions
W_plot = f_W(csi_plot, beta)
plot_test_functions(csi_plot, W_plot)

# Evaluate matrices and vectors

# Afference matrix
from afference import afference_matrix
A = afference_matrix(n_el, dof_el)

from element import Element

el = [Element(dof_el) for n in range(n_el)]

from matrix import element_mass_matrix,element_convection_matrix,element_diffusion_matrix
# Element mass matrix
el = [{} for _ in range(n_el)]
for n in range(n_el):              # (dof_el,n_gauss,N,W,w,J)
    el[n]['M'] = element_mass_matrix(dof_el, n_gauss, N, W, w, J)

# Element convection matrix
for n in range(n_el):
    el[n]['C'] = element_convection_matrix(a, dof_el, n_gauss, dN, W, w, J)

# Element diffusion matrix
for n in range(n_el):
    el[n]['K'] = element_diffusion_matrix(v, dof_el, n_gauss, dN, dW, w, J)


def s_fun(x):
    return 0                          # Source term
s=s_fun(x)*np.ones(x.shape[0])           # Numerical value of source term

# Element load vector
from matrix import element_load_vector
for n in range(n_el):
    el[n]['s'] = s[(n-1)*(dof_el-1):n*(dof_el-1)+1]
    el[n]['f'] = element_load_vector(el[n]['s'], dof_el, n_gauss, N, W, w, J)


# Element abscissae
for n in range(n_el):
    el[n].x=x_i+n*L_el+x_e

# Assemblate matrices and vectors
v_arr = v*np.ones(n_el)
a_arr = a*np.ones(n_el)

# Assemblage of mass matrix
# Element mass matrix
from matrix import element_mass_matrix,element_convection_matrix,element_diffusion_matrix
for n in range(n_el):
    el[n].M=element_mass_matrix(dof_el,n_gauss,N,W,w,J)

# Element convection matrix
for n in range(n_el):
    el[n].C=element_convection_matrix(a_arr[n],dof_el,n_gauss,dN,W,w,J)

# Element diffusion matrix
for n in range(n_el):
    el[n].K=element_diffusion_matrix(v_arr[n],dof_el,n_gauss,dN,dW,w,J)

from matrix import element_load_vector
# Element load vector
for n in range(n_el):
    mlab_n = n+1
    mlab_first = (mlab_n - 1) * (dof_el - 1) + 1
    mlab_last  =       mlab_n * (dof_el - 1) + 1
    el[n].s=s[mlab_first-1:mlab_last]
    el[n].f=element_load_vector(el[n].s,dof_el,n_gauss,N,W,w,J)

# Convection+Diffusion matrix
D=C+K

# Assemblage of load vector
from load_vector import assemble_load_vector
f=assemble_load_vector(el,dof,n_el,dof_el,A)

# Definition of the constrained DOFs
dof_free=dof-len(dof_constrained)
n_dof_constrained=len(dof_constrained)


qq = 0


