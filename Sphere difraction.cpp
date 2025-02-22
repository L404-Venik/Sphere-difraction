#include "Sphere difraction.h"

//std::complex<double> i(0.0, 1.0);
//double PI = std::numbers::pi_v<double>;

extern std::complex<double> i;
extern double PI;
extern double EPS;

#define MatrixType arma::cx_dmat
#define iHO  1 // Hankel order

void calculate_T(int layers_number, double k, std::vector<double>& eps, std::vector<double>& r,
	std::vector<MatrixType>& T_e, std::vector<MatrixType>& T_m)
{
	char N = 50;

	// Define a 4D vectors A and B
	std::vector<std::vector<MatrixType>> A_e(layers_number, std::vector<MatrixType>(N, MatrixType(2, 2, arma::fill::none)));
	std::vector<std::vector<MatrixType>> A_m(layers_number, std::vector<MatrixType>(N, MatrixType(2, 2, arma::fill::none)));
	std::vector<std::vector<MatrixType>> B_e(layers_number, std::vector<MatrixType>(N, MatrixType(2, 2, arma::fill::none)));
	std::vector<std::vector<MatrixType>> B_m(layers_number, std::vector<MatrixType>(N, MatrixType(2, 2, arma::fill::none)));
	//std::vector<std::vector<MatrixType>> vT_e(layers_number, std::vector<MatrixType>(N, MatrixType(2, 2, arma::fill::none)));
	//std::vector<std::vector<MatrixType>> vT_m(layers_number, std::vector<MatrixType>(N, MatrixType(2, 2, arma::fill::none)));
	T_e = std::vector<MatrixType>(N, arma::eye<arma::cx_mat>(2, 2));
	T_m = std::vector<MatrixType>(N, arma::eye<arma::cx_mat>(2, 2));

	for (int j = 0; j < layers_number; j++)
	{
		for (int n = 1; n < N; n++)
		{
			double arg_A = k * std::sqrt(eps[j]) * r[j];

			A_m[j][n](0, 0) = arg_A * std::sph_bessel(n, arg_A);
			A_m[j][n](0, 1) = arg_A * sph_hankel(iHO, n, arg_A);// spherical_hankel(1, n, arg_A);
			A_m[j][n](1, 0) = sqrt(eps[j]) * sph_bessel_derivative(n, arg_A); // with derivative of Bessel function
			A_m[j][n](1, 1) = sqrt(eps[j]) * sph_hankel_derivative(iHO, n, arg_A); // with derivative of Hankel function

			A_e[j][n](0, 0) = eps[j] * A_m[j][n](0, 0);
			A_e[j][n](0, 1) = eps[j] * A_m[j][n](0, 1);
			A_e[j][n](1, 0) = A_m[j][n](1, 0);
			A_e[j][n](1, 1) = A_m[j][n](1, 1);


			double arg_B = k * std::sqrt(eps[j + 1]) * r[j];

			B_m[j][n](0, 0) = arg_B * std::sph_bessel(n, arg_B);
			B_m[j][n](0, 1) = arg_B * sph_hankel(iHO, n, arg_B);
			B_m[j][n](1, 0) = sqrt(eps[j + 1]) * sph_bessel_derivative(n, arg_B); // with derivative of Bessel function
			B_m[j][n](1, 1) = sqrt(eps[j + 1]) * sph_hankel_derivative(iHO, n, arg_B); // with derivative of Hankel function

			B_e[j][n](0, 0) = eps[j + 1] * B_m[j][n](0, 0);
			B_e[j][n](0, 1) = eps[j + 1] * B_m[j][n](0, 1);
			B_e[j][n](1, 0) = B_m[j][n](1, 0);
			B_e[j][n](1, 1) = B_m[j][n](1, 1);

			MatrixType B_m_inv = arma::inv(B_m[j][n]);

			//vT_m[j][n] = B_m_inv * A_m[j][n];
			T_m[n] *= B_m_inv * A_m[j][n];

			MatrixType B_e_inv = arma::inv(B_e[j][n]);
			//vT_e[j][n] = B_e_inv * A_e[j][n];
			T_e[n] *= B_e_inv * A_e[j][n];
		}
	}
}

void calculate_electric_field_close(std::vector<double>& r, std::vector<d_compl>& eps_compl, double lambda)
{
	std::vector<d_compl> vE_r;
	std::vector<d_compl> vE_theta;
	std::vector<d_compl> vE_phi;
	size_t layers_number = r.size() - 1;
	size_t N = 125;
	double k = 2 * PI / lambda; // wave number

	double x = r.back() * 1.1;
	double theta = 0;
	double phi = 0;

	std::vector<double> eps;
	for (d_compl e : eps_compl)
		eps.push_back(e.real());
	eps.push_back(1.0);


	std::vector<MatrixType> T_e;
	std::vector<MatrixType> T_m;
	calculate_T(layers_number, k, eps, r, T_e, T_m);


	for (theta = 0; theta < PI; theta += (PI / 36.0))
	{
		// electric field components
		d_compl E_r = 0;
		d_compl E_theta = 0;
		d_compl E_phi = 0;


		double d_1 = 0;
		double c_1 = 0;
		d_compl R_e = 1, R_m = 1;
		for (int n = 0; n < N; n++)
		{
			std::complex<double> c_n = std::pow(i, (n - 1)) / (k * k) * (2.0 * n + 1) / (n * (n + 1.0));

			std::complex<double> d_e_n = c_n * (T_e[n](1, 0) + T_e[n](1, 1) * R_e) / (T_e[n](0, 0) + T_e[n](0, 1) * R_e);
			std::complex<double> d_m_n = c_n * (T_m[n](1, 0) + T_m[n](1, 1) * R_m) / (T_m[n](0, 0) + T_m[n](0, 1) * R_m);

			double cos_th = cos(theta);
			double sin_th = sin(theta);
			//if (sin_th == 0.0)
			//	sin_th += EPS;
			d_compl Hanc = sph_hankel(iHO, n, k * x);
			d_compl Hanc_der = sph_hankel_derivative(iHO, n, k * x);
			double Leg = std::legendre(n, cos_th);
			double Leg_der = legendre_derivative(n, cos_th);

			E_r += d_e_n * (n * (n + 1) / (x * x)) * Hanc * Leg * cos(phi);

			E_theta = E_theta + (d_e_n * Hanc_der * Leg_der * sin_th) + (i * d_m_n / sin_th * Hanc * Leg);

			E_phi = E_phi + (d_e_n / sin_th * Hanc_der * Leg) + (i * d_m_n * Hanc * Leg_der * sin_th);

			R_e = d_e_n / c_n;
			R_m = d_m_n / c_n;
		}

		E_theta *= k / x * cos(phi);
		E_phi *= -k / x * sin(phi);

		vE_r.push_back(E_r);
		vE_theta.push_back(E_theta);
		vE_phi.push_back(E_phi);
	}

	//writeNumericVectorToFile("E_theta.txt", vE_theta);
}


void calculate_electric_field_far(std::vector<double>& r, std::vector<d_compl>& eps_compl, double lambda)
{
	std::vector<d_compl> vE_theta;
	std::vector<d_compl> vE_phi;
	size_t layers_number = r.size() - 1;
	size_t N = 25;
	double k = 2 * PI / lambda; // wave number

	double x = r.back() * 10;
	double theta = 0;
	double phi = 0;

	std::vector<double> eps;
	for (d_compl e : eps_compl)
		eps.push_back(e.real());
	eps.push_back(1.0);


	std::vector<MatrixType> T_e;
	std::vector<MatrixType> T_m;
	calculate_T(layers_number, k, eps, r, T_e, T_m);


	for (theta = 0; theta < 2 * PI; theta += (PI / (3 * 36.0)))
	{
		// electric field components
		d_compl E_th = 0; // E theta
		d_compl E_ph = 0; // E phi
		d_compl S_th = 0; // S theta
		d_compl S_ph = 0; // S phi

		// constants for this step
		double cos_th = cos(theta);
		double sin_th = sin(theta);


		d_compl R_e = 1, R_m = 1;
		d_compl first_term, second_term;
		d_compl d_e_n, d_m_n, c_n;
		double arg_R = k * std::sqrt(eps[1]) * r[0];
		d_compl i_pow = -i;
		for (int n = 1; n < N; n++)
		{
			c_n = std::pow(i, (n - 1)) / (k * k) * (2.0 * n + 1) / (n * (n + 1.0));

			R_m = -std::sph_bessel(n, arg_R) / sph_hankel(iHO, n, arg_R);
			R_e = -sph_bessel_derivative(n, arg_R) / sph_hankel_derivative(iHO, n, arg_R);
			d_e_n = c_n * (T_e[n](1, 0) + T_e[n](1, 1) * R_e) / (T_e[n](0, 0) + T_e[n](0, 1) * R_e);
			d_m_n = c_n * (T_m[n](1, 0) + T_m[n](1, 1) * R_m) / (T_m[n](0, 0) + T_m[n](0, 1) * R_m);


			first_term = std::assoc_legendre(n, 1, cos_th) / sin_th;
			second_term = assoc_legendre_derivative(n, 1, cos_th) * sin_th;

			if (std::isnan(first_term.real()) || std::isinf(first_term.real()) || std::isnan(first_term.imag()) || std::isinf(first_term.imag()))
				first_term = 0.0;
			if (std::isnan(second_term.real()) || std::isinf(second_term.real()) || std::isnan(second_term.imag()) || std::isinf(second_term.imag()))
				second_term = 0.0;

			S_th += i_pow * (d_m_n * first_term - d_e_n * second_term);

			S_ph += i_pow * (d_m_n * second_term - d_e_n * first_term);

			i_pow *= -i;
		}

		S_th *= k * k;
		S_ph *= k * k;

		E_th = S_th * (std::exp(i * k * x) * std::cos(phi) / (k * x));
		E_ph = S_ph * (std::exp(i * k * x) * std::sin(phi) / (k * x));

		vE_theta.push_back(E_th);
		vE_phi.push_back(E_ph);
	}

	int E_size = vE_theta.size();

	vE_theta[0] = vE_theta[1];
	vE_theta[E_size - 1] = vE_theta[E_size - 2];
	vE_theta[E_size / 2] = vE_theta[E_size / 2 - 1];

	writeNumericVectorToFile("E_theta.txt", vE_theta);
	//writeNumericVectorToFile("E_phi.txt", vE_phi);
}


void calculate_electric_field_classical(std::vector<double>& r, std::vector<d_compl>& eps_compl, double lambda)
{

}

void FirstExercise()
{
	double lambda = 2 * PI / 1000.0; // wave length
	double k = 2 * PI / lambda; // wave number
	int layers_number = 3;
	std::vector<d_compl> eps(layers_number + 1);
	std::vector<double> r(layers_number + 1);

	eps = { 1, d_compl(1.1, 0.01), 2.5 , 8.85e-12 };
	r = { 0.01, 0.01 + PI / k, 0.01 + 0.5 / k };
	calculate_electric_field_far(r, eps, lambda);
}

void SimpleSphere()
{
	double lambda = 2 * PI / 1000.0; // wave length
	double k = 2 * PI / lambda; // wave number
	int layers_number = 1;
	std::vector<d_compl> eps(layers_number + 1);
	std::vector<double> r(layers_number + 1);

	eps = { 1, 8.85e-12 };
	r = { 0.01, 0.1 };
	calculate_electric_field_far(r, eps, lambda);
}

int main()
{
	FirstExercise();
	//SimpleSphere();
}
