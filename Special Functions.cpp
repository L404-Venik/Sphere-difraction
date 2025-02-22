#include "Special Functions.h"

std::complex<double> i(0.0, 1.0);
double PI = std::numbers::pi_v<double>;
double EPS = 0.000000001;

// Hackel
std::complex<double> sph_hankel(char order, double nu, double x)
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

std::complex<double> sph_hankel_derivative(char order, double n, double arg)
{
	std::complex<double> result;
	std::complex<double> one = sph_bessel_derivative(n, arg);
	std::complex<double> two = sph_neumann_derivative(n, arg);

	if (order == 1)
		result = one + i * two;
	else if (order == 2)
		result = one - i * two;
	else
		assert(false); // wrong order

	return result;
}

// Bessel
double bessel_derivative(double n, double arg)
{
	return (1.0 / 2.0) * (std::cyl_bessel_j(n - 1, arg) - std::cyl_bessel_j(n + 1, arg));
}

double sph_bessel_derivative(double n, double arg)
{
	assert(arg != 0);
	double result;
	/*result = -1.0 / 2.0 * std::sqrt(PI / (2 * std::pow(arg, 3)));
	result *= std::cyl_bessel_j(n + 1.0 / 2.0, arg);
	result += std::sqrt(PI / (2 * arg)) * bessel_derivative(n + 1.0 / 2.0, arg);*/
	result = std::sph_bessel(n, arg) + arg * (std::sph_bessel(n - 1, arg) - std::sph_bessel(n + 1, arg));
	result *= 0.5;

	return result;
}

// Neumann
double neumann_derivative(double n, double arg)
{
	assert(sin(n * PI) != 0);
	double result;

	result = bessel_derivative(n, arg) * cos(n * PI) - bessel_derivative(-n, arg);
	result /= sin(n * PI);

	return result;
}

double sph_neumann_derivative(double n, double arg)
{
	assert(arg != 0);
	double result;

	//result = -1.0 / 2.0 * std::sqrt(PI / (2 * std::pow(arg, 3))) * std::cyl_neumann(n + 1.0 / 2.0, arg)
	//	+ std::sqrt(PI / (2 * arg)) * neumann_derivative(n + 1.0 / 2.0, arg);
	result = std::sph_neumann(n, arg) + arg * (std::sph_neumann(n - 1, arg) - std::sph_neumann(n + 1, arg));
	result *= 0.5;

	return result;
}


// Legendre
double legendre_derivative(double n, double arg)
{
	//assert(arg != 1);
	double result;
	double P1 = std::legendre(n - 1, arg);
	double P2 = std::legendre(n, arg);

	if (arg - 1.0 < EPS)
		arg += EPS;

	result = n / (1 - arg * arg) * (P1 - arg * P2);

	return result;
}

double assoc_legendre_derivative(unsigned int n, unsigned int m, double arg)
{
	double result;
	double P1 = std::assoc_legendre(n - 1, m, arg);
	double P2 = std::assoc_legendre(n, m, arg);

	if (arg - 1.0 < EPS)
		arg += EPS;

	result = n / (1 - arg * arg) * (P1 - arg * P2);

	return result;

}