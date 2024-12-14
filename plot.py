import numpy as np
import matplotlib.pyplot as plt


def plot_shape_functions(csi, N):
    # Plot of shape functions
    c = len(N)
    col = ['r', 'b', 'g']
    leg = [f'N_{n + 1} (ξ)' for n in range(c)]

    plt.figure(facecolor='white')
    plt.gca().tick_params(labelsize=14)

    for n in range(c):
        plt.plot(csi, N[n], col[n], linewidth=3)

    plt.plot(csi, np.zeros(len(csi)), 'k', linewidth=3)
    plt.title('Shape functions', fontsize=14)
    plt.xlabel('ξ', fontsize=14)
    plt.ylabel('N', fontsize=14)
    plt.legend(leg)
    plt.grid(True, which='both')
    plt.xlim([-1.1, +1.1])
    plt.ylim([-0.5, +1.5])
    plt.show()
    plt.savefig('shape_functions.png')


import numpy as np
import matplotlib.pyplot as plt


def plot_test_functions(csi, W):
    # Plot of test functions
    c = W.shape[1]
    col = ['r', 'b', 'g']
    leg = [f'W_{n + 1} (ξ)' for n in range(c)]

    plt.figure(facecolor='white')
    plt.gca().tick_params(labelsize=14)

    for n in range(c):
        plt.plot(csi, W[:, n], col[n], linewidth=3)

    plt.plot(csi, np.zeros(len(csi)), 'k', linewidth=3)

    plt.title('Test functions', fontsize=14)
    plt.xlabel('ξ', fontsize=14)
    plt.ylabel('W', fontsize=14)
    plt.legend(leg)
    plt.grid(True, which='both')
    plt.xlim([-1.1, 1.1])
    plt.ylim([-0.5, 1.5])
    plt.show()
    plt.savefig('test_functions.png')