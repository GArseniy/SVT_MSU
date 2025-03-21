import matplotlib.pyplot as plt
import numpy as np
import csv
import os


def tridiagonal_matrix_algorithm(a, b, c, d):
    """Solves a system of equations using the tridiagonal matrix algorithm (TDMA).

    Args:
        a: Lower diagonal (n-1 elements).
        b: Main diagonal (n elements).
        c: Upper diagonal (n-1 elements).
        d: Right-hand side vector (n elements).

    Returns:
        The solution to the system (vector x).
    """
    n = len(b)
    x = np.zeros(n)  

    alpha = np.zeros(n)
    beta = np.zeros(n)

    alpha[0] = -c[0] / b[0]
    beta[0] = d[0] / b[0]

    for i in range(1, n):
        denominator = b[i] + a[i] * alpha[i-1]
        alpha[i] = -c[i] / denominator
        beta[i] = (d[i] - a[i] * beta[i-1]) / denominator

    x[n-1] = beta[n-1]

    for i in range(n-2, -1, -1):
        x[i] = alpha[i] * x[i+1] + beta[i]

    return x


def extend_vec(x):
    v = np.zeros(len(x) + 2)
    v[1:-1] = x 
    return v


def chebyshev_distance(vector1, vector2):
    return np.max(np.abs(vector1 - vector2))


def l2_distance(vector1, vector2):
    return np.sqrt(np.sum((vector1 - vector2) ** 2))


def plot_function_and_vector(func, vector, num_points=100, filename="plot/sol.png"):
    x = np.linspace(0, 1, num_points)  
    y = func(x)                       
    
    x_vector = np.linspace(0, 1, len(vector)) 

    plt.figure(figsize=(10, 6)) 

    plt.plot(x, y, label="anal sol: sin(pi*x)", color="blue")  

    plt.plot(x_vector, vector, 'ro', label="approx sol", markersize=0.005)  

    plt.title("Solution Laplace Dirichlet problem")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.legend()

    plt.grid(True)

    plt.savefig(filename)

    plt.close()


def plot_conv(vector, max_pow, filename="plot/error.png"):
                       
    x_vector = np.array([10 ** i for i in range(1, max_pow + 1)])

    plt.figure(figsize=(10, 6)) 

    plt.plot(x_vector, vector, label="Convergence", color="blue")  

    plt.yscale('log')
    plt.xscale('log')

    plt.title("Convergence plot")
    plt.xlabel("N")
    plt.ylabel("error")

    plt.legend()

    plt.grid(True)

    plt.savefig(filename)

    plt.close()


def save_error_csv(array, filename="data/error.csv"):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        powers_of_ten = [10**i for i in range(1, len(array) + 1)]

        for power, value in zip(powers_of_ten, array):
            writer.writerow([power, value])



if __name__ == '__main__':
    if not os.path.exists('plot'): 
        os.mkdir('./plot')

    if not os.path.exists('data'):
        os.mkdir('./data')

    f = lambda x: np.pi * np.pi * np.sin(np.pi * x)
    u = lambda x: np.sin(np.pi * x)

    max_pow = 5

    normas_vec_c = np.zeros(max_pow)
    normas_vec_l2 = np.zeros(max_pow)
    normas_iter = 0

    for N in [10 ** i for i in range(1, max_pow + 1)]:

        a = np.full(N - 1, -1)
        b = np.full(N - 1, +2)
        c = np.full(N - 1, -1)

        d = f(np.linspace(0, 1, N + 1)[1:-1]) / N / N

        x = tridiagonal_matrix_algorithm(a, b, c, d)

        x = extend_vec(x)

        cor_x = u(np.linspace(0, 1, N + 1))

        plot_function_and_vector(u, x)

        normas_vec_l2[normas_iter] = l2_distance(x, cor_x)
        normas_vec_c[normas_iter] = chebyshev_distance(x, cor_x)
        normas_iter += 1

    plot_conv(normas_vec_c, max_pow, "plot/error_c.png")
    plot_conv(normas_vec_l2, max_pow, "plot/error_l2.png")

    save_error_csv(normas_vec_c, "data/c.csv")
    save_error_csv(normas_vec_l2, "data/l2.csv")
