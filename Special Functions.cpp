#include "Special Functions.h"

std::complex<double> i(0.0, 1.0);
double PI = std::numbers::pi_v<double>;

// Hackel
std::complex<double> spherical_hankel(char order, double nu, double x)
{
	double J = std::sph_bessel(nu, x);
	double N = std::sph_neumann(nu, x);

	std::complex<double> result;
	if (order == 1)
		result = J + i * N;
	else if (order == 2)
		result = J - i * N;
	else
		assert(false); // wrong order

	return result;
}

std::complex<double> normalized_hancel_derivative(char order, double n, double arg)
{
	std::complex<double> result;
	std::complex<double> one = normalized_bessel_derivative(n, arg);
	std::complex<double> two = normalized_neumann_derivative(n, arg);

	if (order == 1)
		result = normalized_bessel_derivative(n, arg) + i * normalized_neumann_derivative(n, arg);
	else if (order == 2)
		result = normalized_bessel_derivative(n, arg) - i * normalized_neumann_derivative(n, arg);
	else
		assert(false); // wrong order

	return result;
}

// Bessel
double bessel_derivative(double n, double arg)
{
	return (1.0 / 2.0) * (std::cyl_bessel_j(n - 1, arg) - std::cyl_bessel_j(n + 1, arg));
}

double normalized_bessel_derivative(double n, double arg)
{
	assert(arg != 0);
	double result = -1.0 / 2.0 * std::sqrt(PI / (2 * std::pow(arg, 3)));
	result *= std::cyl_bessel_j(n + 1.0 / 2.0, arg);
	result += std::sqrt(PI / (2 * arg)) * bessel_derivative(n + 1.0 / 2.0, arg);
	return result;
}

// Neumann
double neumann_derivative(double n, double arg)
{
	double result;

	result = bessel_derivative(n, arg) * cos(n * PI) - bessel_derivative(-n, arg);
	result /= sin(n * PI);

	return result;
}

double normalized_neumann_derivative(double n, double arg)
{
	double result;

	result = -1.0 / 2.0 * std::sqrt(PI / (2 * std::pow(arg, 3))) * std::cyl_neumann(n + 1.0 / 2.0, arg)
		+ std::sqrt(PI / (2 * arg)) * neumann_derivative(n + 1.0 / 2.0, arg);

	return result;
}


// Legendre
double legendre_derivative(double n, double arg)
{
	double result;
	double P1 = std::legendre(n - 1, arg);
	double P2 = std::legendre(n, arg);

	result = n / (1 - arg * arg) * (P1 - arg * P2);

	return result;
}