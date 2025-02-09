#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <complex>
#include <numbers>
#include <cassert>
#include <armadillo>


// Hancel
std::complex<double> spherical_hankel(char order, double nu, double x);

std::complex<double> normalized_hancel_derivative(char order, double n, double arg);


// Bessel
double bessel_derivative(double n, double arg);

double normalized_bessel_derivative(double n, double arg);

// Neumann
double neumann_derivative(double n, double arg);

double normalized_neumann_derivative(double n, double arg);

// Legendre
double legendre_derivative(double n, double arg);