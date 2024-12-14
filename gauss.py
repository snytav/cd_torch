import numpy as np

def Gauss_parameters(n):
    # Parameters for Gauss integration rule in [-1,+1]
    if n == 1:  # Gauss with 1 node
        csi = [0]
        w = [2]
    elif n == 2:  # Gauss with 2 nodes
        csi = [-1.0/np.sqrt(3), +1.0/np.sqrt(3)]
        w = [1, 1]
    elif n == 3:  # Gauss with 3 nodes
        csi = [-np.sqrt(3.0/5), 0, +np.sqrt(3.0/5)]
        w = [5.0/9, 8.0/9, 5.0/9]
    elif n == 4:  # Gauss with 4 nodes
        csi = [
            -1.0/35 * np.sqrt(525 + 70 * np.sqrt(30)),
            -1.0/35 * np.sqrt(525 - 70 * np.sqrt(30)),
            +1.0/35 * np.sqrt(525 - 70 * np.sqrt(30)),
            +1.0/35 * np.sqrt(525 + 70 * np.sqrt(30))
        ]
        w = [
            1.0/36 * (18 - np.sqrt(30)),
            1.0/36 * (18 + np.sqrt(30)),
            1.0/36 * (18 + np.sqrt(30)),
            1.0/36 * (18 - np.sqrt(30))
        ]
    return csi, w

def f_N(csi):
    # Shape functions
    N1 = 1.0/2 * (1 - csi)
    N2 = 1.0/2 * (1 + csi)
    N = [N1, N2]
    return N

def f_dN(csi):
    # 1st derivatives of shape functions
    dN1 = -1.0/2
    dN2 = +1.0/2
    dN = [dN1, dN2]
    return dN

def shape_functions_gauss_points(csi):
    # Computation of shape functions (and derivatives) at Gauss points
    n_gauss = len(csi)
    N = []
    dN = []
    for n in range(n_gauss):
        N.append(f_N(csi[n]))
        dN.append(f_dN(csi[n]))
    return N, dN

def f_W(csi, beta):
    W1 = 1.0/2 * (1 - csi) - 3.0/4 * beta * (1 - csi**2)
    W2 = 1.0/2 * (1 + csi) + 3.0/4 * beta * (1 - csi**2)
    W = np.column_stack((W1, W2))
    return W

def f_W(csi, beta):
    W1 = 1.0/2 * (1 - csi) - 3.0/4 * beta * (1 - csi**2)
    W2 = 1.0/2 * (1 + csi) + 3.0/4 * beta * (1 - csi**2)
    W = np.column_stack((W1, W2))
    return W

def f_dW(csi, beta):
    # 1st derivatives of test functions
    dW1 = -1.0/2 + 3.0/2 * beta * csi
    dW2 = 1.0/2 - 3.0/2 * beta * csi
    dW = np.column_stack((dW1, dW2))
    return dW


def test_functions_Gauss_points(csi, beta):
    # Computation of test functions (and derivatives) at Gauss points
    n_gauss = len(csi)
    W = []
    dW = []
    for n in range(n_gauss):
        W.append(f_W(csi[n], beta))
        dW.append(f_dW(csi[n], beta))
    return W, dW