def plot_geometry(x, h, n_np, L, n_el, n_gauss, L_el, x_gauss):
    import matplotlib.pyplot as plt

    # Plot of geometry
    plt.figure(facecolor='white')
    plt.gca().tick_params(labelsize=14)
    plt.plot(x, [0] * len(x), 'k', linewidth=3)
   # plt.hold(True)

    for n in range(1, n_np + 1):
        plt.plot([x[0] + h * (n - 1), x[0] + h * (n - 1)], [-L / 20, L / 20], 'k', linewidth=2)
        plt.text(x[0] + h * (n - 1) - h / 15, -L / 10, str(n))

    for i in range(1, n_el + 1):
        for n in range(1, n_gauss + 1):
            plt.plot([x[0] + (i - 1) * L_el + x_gauss[n - 1], x[0] + (i - 1) * L_el + x_gauss[n - 1]],
                     [-L / 30, L / 30], 'r', linewidth=2)

    #plt.hold(False)
    plt.title('Geometry', fontsize=14)
    plt.xlabel('x [m]', fontsize=14)
    plt.ylabel('y [m]', fontsize=14)
    #plt.grid(True)
    plt.minorticks_on()
    plt.xlim([x[0] - L / 10, x[-1] + L / 10])
    plt.ylim([-L, L])
    plt.show()
    plt.savefig('geometry.png')