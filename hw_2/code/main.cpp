#include "inmost.h"
#include <iostream>
#include <functional>
#include <cmath>
#include <chrono>
#include <vector>


using namespace INMOST;

Sparse::Vector
my_solver(std::function<double(double, double)> f,
            std::function<double(double)> g_bottom,
            std::function<double(double)> g_top,
            std::function<double(double)> g_left,
            std::function<double(double)> g_right, size_t n)
{
    double h = 1.0 / n;
    
    Sparse::Matrix A;
    Sparse::Vector b;
    Sparse::Vector x;
    
    A.SetInterval(0, (n - 1) * (n - 1));
    b.SetInterval(0, (n - 1) * (n - 1));
    x.SetInterval(0, (n - 1) * (n - 1));


    std::function<void(size_t, size_t, size_t)> set_coeff =
        [&] (size_t row, size_t i_set, size_t j_set) {
            if (i_set == 0) b[row] += g_bottom(j_set * h) / h / h;
            else if (i_set == n) b[row] += g_top(j_set * h) / h / h;
            else if (j_set == 0) b[row] += g_left(i_set * h) / h / h;
            else if (j_set == n) b[row] += g_right(i_set * h) / h / h;
            else A[row][(i_set - 1) * (n - 1) + j_set - 1] = -1.0;
        };

    for (size_t i = 1; i < n; ++i) {
        for (size_t j = 1; j < n; ++j) {
            size_t idx = (i - 1) * (n - 1) + j - 1;
            A[idx][idx] = 4.0;
            b[idx] = f(i * h, j * h);
            set_coeff(idx, i - 1, j);
            set_coeff(idx, i + 1, j);
            set_coeff(idx, i, j - 1);
            set_coeff(idx, i, j + 1);
        }
    }

    Solver S(Solver::INNER_MPTILUC);
    S.SetParameter("absolute_tolerance", "1e-12");
    S.SetParameter("relative_tolerance", "1e-12");
    S.SetParameter("drop_tolerance", "0.005");
    S.SetMatrix(A);
    S.Solve(b, x);
    
    std::cout << S.Iterations() << ' ' << S.IterationsTime() << ' ';

    for (size_t i = 0; i < (n - 1) * (n - 1); ++i) x[i] *= h * h;
    
    return x;
}


double u(double x, double y) { return sin(x) * sin(5 * y); }


double f(double x, double y) { return 26.0 * u(x, y); }


double norm_L2_01(Sparse::Vector& x, std::function<double(double, double)> u, size_t n)
{
    double h = 1.0 / n;
    double norm = 0.0;
    for (size_t i = 1; i < n; i++) {
        for (size_t j = 1; j < n; j++) {
            size_t idx = (i - 1) * (n - 1) + j - 1;
            double diff = u(i * h, j * h) - x[idx];
            norm += diff * diff;
        }
    }
    return sqrt(norm);
}

double norm_C_01(Sparse::Vector& x, std::function<double(double, double)> u, size_t n)
{
    double h = 1.0 / n;
    double norm = 0.0;
    for (size_t i = 1; i < n; i++) {
        for (size_t j = 1; j < n; j++) {
            size_t idx = (i - 1) * (n - 1) + j - 1;
            norm = std::max(norm, abs(u(i * h, j * h) - x[idx]));
        }
    }
    return sqrt(norm);
}

int main(int argc, char *argv[]) {
    size_t max_n = 256;

    for (size_t n = 2; n <= max_n; n *= 2) {
        std::cout << n << ' ';
        auto start = std::chrono::steady_clock::now();
        Sparse::Vector x = my_solver(f,
                    [] (double y) { return u(0.0, y); },
                    [] (double y) { return u(1.0, y); },
                    [] (double x) { return u(x, 0.0); },
                    [] (double x) { return u(x, 1.0); }, n);
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> time = end - start;
        
        std::cout << time.count() << ' ' << norm_L2_01(x, u, n) << ' ' << norm_C_01(x, u, n) << std::endl;
        
    }
    return 0;
}