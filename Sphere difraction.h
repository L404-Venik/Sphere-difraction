#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <complex>
#include <numbers>
#include <cassert>
#include <armadillo>

using d_compl = std::complex<double>;

double PI = std::numbers::pi_v<double>;
double EPS = 0.000000001;
std::complex<double> i(0.0, 1.0);

// Hancel
std::complex<double> spherical_hankel(char order, double nu, double x);

d_compl normalized_hancel_derivative(char order, double n, double arg);


// Bessel
double bessel_derivative(double n, double arg);

double normalized_bessel_derivative(double n, double arg);

// Neumann
double neumann_derivative(double n, double arg);

double normalized_neumann_derivative(double n, double arg);

// Legendre
double legendre_derivative(double n, double arg);