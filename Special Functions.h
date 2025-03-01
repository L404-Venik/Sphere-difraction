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
std::complex<double> sph_hankel(char order, double nu, double x);

std::complex<double> sph_hankel_derivative(char order, double n, double arg);


// Bessel
double bessel_derivative(double n, double arg);

double sph_bessel_derivative(unsigned int n, double arg);

// Neumann
double neumann_derivative(double n, double arg);

double sph_neumann_derivative(unsigned int n, double arg);

// Legendre
double legendre_derivative(double n, double arg);

double assoc_legendre_derivative(unsigned int n, unsigned int m, double arg);