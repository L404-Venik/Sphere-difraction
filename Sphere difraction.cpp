#include "Sphere difraction.h"

//std::complex<double> i(0.0, 1.0);
//double PI = std::numbers::pi_v<double>;

extern std::complex<double> i;
extern double PI;

#define MatrixType arma::cx_dmat

void calculate_electric_field(std::vector<double>& r, std::vector<d_compl>& eps_compl, double lambda)
{
	std::vector<d_compl> vE_r;
	std::vector<d_compl> vE_theta;
	std::vector<d_compl> vE_phi;
	size_t layers_number = r.size();
	size_t N = 125;
	double k = 2 * PI / lambda; // wave number

	double x = r.back() * 1.1;
	double theta = 0;
	double phi = 0;

	std::vector<double> eps;
	for (d_compl e : eps_compl)
		eps.push_back(e.real());
	eps.push_back(1.0);

	for (theta = 0; theta < PI; theta += (PI / 36.0))
	{
		// electric field components
		d_compl E_r = 0;
		d_compl E_theta = 0;
		d_compl E_phi = 0;

		// Define a 4D vectors A and B
		std::vector<std::vector<MatrixType>> A_e(layers_number, std::vector<MatrixType>(N, MatrixType(2, 2, arma::fill::none)));
		std::vector<std::vector<MatrixType>> A_m(layers_number, std::vector<MatrixType>(N, MatrixType(2, 2, arma::fill::none)));
		std::vector<std::vector<MatrixType>> B_e(layers_number, std::vector<MatrixType>(N, MatrixType(2, 2, arma::fill::none)));
		std::vector<std::vector<MatrixType>> B_m(layers_number, std::vector<MatrixType>(N, MatrixType(2, 2, arma::fill::none)));
		std::vector<std::vector<MatrixType>> vT_e(layers_number, std::vector<MatrixType>(N, MatrixType(2, 2, arma::fill::none)));
		std::vector<std::vector<MatrixType>> vT_m(layers_number, std::vector<MatrixType>(N, MatrixType(2, 2, arma::fill::none)));
		std::vector<MatrixType> T_e(N, arma::eye<arma::cx_mat>(2, 2));
		std::vector<MatrixType> T_m(N, arma::eye<arma::cx_mat>(2, 2));

		for (int j = 0; j < layers_number; j++)
		{
			for (int n = 0; n < N; n++)
			{
				double arg_A = k * std::sqrt(eps[j]) * r[j];

				A_m[j][n](0, 0) = std::sph_bessel(n, arg_A);
				A_m[j][n](0, 1) = spherical_hankel(1, n, arg_A);
				A_m[j][n](1, 0) = sqrt(eps[j]) * normalized_bessel_derivative(n, arg_A); // with derivative of Bessel function
				A_m[j][n](1, 1) = sqrt(eps[j]) * normalized_hancel_derivative(1, n, arg_A); // with derivative of Hankel function

				A_e[j][n](0, 0) = eps[j] * A_m[j][n](0, 0);
				A_e[j][n](0, 1) = eps[j] * A_m[j][n](0, 1);
				A_e[j][n](1, 0) = A_m[j][n](1, 0);
				A_e[j][n](1, 1) = A_m[j][n](1, 1);


				double arg_B = k * std::sqrt(eps[j + 1]) * r[j];

				B_m[j][n](0, 0) = std::sph_bessel(n, arg_B);
				B_m[j][n](0, 1) = spherical_hankel(1, n, arg_B);
				B_m[j][n](1, 0) = sqrt(eps[j + 1]) * normalized_bessel_derivative(n, arg_B); // with derivative of Bessel function
				B_m[j][n](1, 1) = sqrt(eps[j + 1]) * normalized_hancel_derivative(1, n, arg_B); // with derivative of Hankel function

				B_e[j][n](0, 0) = eps[j + 1] * B_m[j][n](0, 0);
				B_e[j][n](0, 1) = eps[j + 1] * B_m[j][n](0, 1);
				B_e[j][n](1, 0) = B_m[j][n](1, 0);
				B_e[j][n](1, 1) = B_m[j][n](1, 1);

				MatrixType B_m_inv = arma::inv(B_m[j][n]);
				//d_compl a = B_m_inv(0, 0) * A_m[j][n](0, 0) + B_m_inv(0, 1) * A_m[j][n](1, 0);
				//d_compl b = B_m_inv(0, 0) * A_m[j][n](0, 1) + B_m_inv(0, 1) * A_m[j][n](1, 1);
				//d_compl c = B_m_inv(1, 0) * A_m[j][n](0, 0) + B_m_inv(1, 1) * A_m[j][n](1, 0);
				//d_compl d = B_m_inv(1, 0) * A_m[j][n](0, 1) + B_m_inv(1, 1) * A_m[j][n](1, 1);

				vT_m[j][n] = B_m_inv * A_m[j][n];
				T_m[n] *= vT_m[j][n];

				MatrixType B_e_inv = arma::inv(B_e[j][n]);
				vT_e[j][n] = B_e_inv * A_e[j][n];
				T_e[n] *= vT_e[j][n];
			}
		}


		double d_1 = 0;
		double c_1 = 0;
		d_compl R_e = 1, R_m = 1;
		for (int n = 1; n < N; n++)
		{
			std::complex<double> c_n = std::pow(i, (n - 1)) / (k * k) * (2.0 * n + 1) / (n * (n + 1.0));

			std::complex<double> d_e_n = c_n * (T_e[n](1, 0) + T_e[n](1, 1) * R_e) / (T_e[n](0, 0) + T_e[n](0, 1) * R_e);
			std::complex<double> d_m_n = c_n * (T_m[n](1, 0) + T_m[n](1, 1) * R_m) / (T_m[n](0, 0) + T_m[n](0, 1) * R_m);

			double cos_th = cos(theta);
			double sin_th = sin(theta);
			if (sin_th == 0.0)
				sin_th += EPS;
			d_compl Hanc = spherical_hankel(1, n, k * x);
			d_compl Hanc_der = normalized_hancel_derivative(1, n, k * x);
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

	writeComplexVectorsToFile("results.txt", vE_r, vE_theta, vE_phi);
	//assert(false);
}

void FirstExercise()
{
	double lambda = 2 * PI / 1000.0; // wave length
	double k = 2 * PI / lambda; // wave number
	int layers_number = 3;
	std::vector<d_compl> eps(layers_number + 1);
	std::vector<double> r(layers_number + 1);

	eps = { 1, d_compl(1.1, 0.01), 2.5 };
	r = { 0.01, 0.01 + PI / k, 0.01 + 0.5 / k };
	calculate_electric_field(r, eps, lambda);
}

void TestSpecialFunctions()
{
	std::vector<double> v;
	for (int i = 1; i < 1000; i++)
		v.push_back(normalized_bessel_derivative(0, 0.1 * i));

	writeNumericVectorToFile("bessels.txt", v);
}

int main()
{
	FirstExercise();
}
