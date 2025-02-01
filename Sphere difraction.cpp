#include "Sphere difraction.h"

double PI = std::numbers::pi_v<double>;
std::complex<double> i = (0, 1);

std::complex<double> spherical_hankel_first_order(double nu, double x) 
{
    double J_1 = std::cyl_bessel_j(-nu, x);
    double J_2 = std::cyl_bessel_j(nu, x);
    std::complex<double> result;

    result = J_1 - std::exp(-nu * PI * i) * J_2;
    result /= -i * sin(nu * PI);
    
    return result;
}

std::complex<double> spherical_hankel_first_order_dirivative(double nu, double x) 
{
    assert(false); //unfinished
    double J_1 = std::cyl_bessel_j(-nu, x);
    double J_2 = std::cyl_bessel_j(nu, x);
    std::complex<double> result;

    result = J_1 - std::exp(-nu * PI * i) * J_2;
    result /= -i * sin(nu * PI);
    
    return result;
}

d_compl normalized_hancel_dirivative(double n, double arg)
{
    assert(false); //unfinished, производная берётся от сложной функции
    double normalizer = std::sqrt(PI / (2 * arg));
    return normalizer * spherical_hankel_first_order_dirivative(n + 1 / 2, arg);
}

d_compl normalized_hancel(double n, double arg)
{
    double normalizer = std::sqrt(PI / (2 * arg));
    return normalizer * spherical_hankel_first_order(n + 1 / 2, arg);
}

double normalized_bessel(double n, double arg)
{
    double normalizer = std::sqrt(PI / (2 * arg));
    return normalizer * std::cyl_bessel_j(n + 1 / 2, arg);
}

int main()
{
    // electric field components
    d_compl E_r = 0;
    d_compl E_theta = 0;
    d_compl E_phi = 0;

    int layers_number = 1;
    int N = 10;
    std::vector<double> eps(layers_number + 1, 1);
    std::vector<double> r(layers_number + 1, 1);

    double lambda = 0.001; // wave length
    double k = 2 * PI / lambda; // wave number

    double x = 0.1;
    double theta = 0;
    double phi = 0;



    // Define a 4D vectors A and B
    std::vector<std::vector<arma::cx_dmat>> A_e(layers_number, std::vector<arma::cx_dmat>(N, arma::cx_dmat(2, 2, arma::fill::none)));
    std::vector<std::vector<arma::cx_dmat>> A_m(layers_number, std::vector<arma::cx_dmat>(N, arma::cx_dmat(2, 2, arma::fill::none)));
    std::vector<std::vector<arma::cx_dmat>> B_e(layers_number, std::vector<arma::cx_dmat>(N, arma::cx_dmat(2, 2, arma::fill::none)));
    std::vector<std::vector<arma::cx_dmat>> B_m(layers_number, std::vector<arma::cx_dmat>(N, arma::cx_dmat(2, 2, arma::fill::none)));
    //std::vector<std::vector<arma::cx_dmat>> T_e(layers_number, std::vector<arma::cx_dmat>(N, arma::cx_dmat(2, 2, arma::fill::none)));
    //std::vector<std::vector<arma::cx_dmat>> T_m(layers_number, std::vector<arma::cx_dmat>(N, arma::cx_dmat(2, 2, arma::fill::none)));
    std::vector<arma::cx_dmat> T_e( N, arma::cx_dmat(2, 2, arma::fill::ones)); // ?
    std::vector<arma::cx_dmat> T_m( N, arma::cx_dmat(2, 2, arma::fill::ones)); // ?

    for (int j = 0; j < layers_number; j ++)
    {
        for (int n = 0; n < N; n++)
        {
            double arg_A = k * std::sqrt(eps[j]) * r[j];

            A_m[j][n](0, 0) = normalized_bessel(n, arg_A);
            A_m[j][n](0, 1) = normalized_hancel(n, arg_A);
            A_m[j][n](0, 1) = sqrt(eps[j]) * (1 / 2) * (normalized_bessel(n - 1, arg_A) - normalized_bessel(n + 1, arg_A)); // with derivative of Bessel function
            A_m[j][n](1, 1) = sqrt(eps[j]) * normalized_hancel_dirivative(n, arg_A); // with derivative of Hankel function

            A_e[j][n](0, 0) = eps[j] * A_m[j][n](0, 0);
            A_e[j][n](0, 1) = eps[j] * A_m[j][n](0, 1);
            A_e[j][n](0, 1) = A_m[j][n](0, 1);
            A_e[j][n](1, 1) = A_m[j][n](1, 1);


            double arg_B = k * std::sqrt(eps[j + 1]) * r[j];

            B_m[j][n](0,0) = normalized_bessel(n, arg_A);
            B_m[j][n](0,1) = normalized_hancel(n, arg_A);
            B_m[j][n](1,0) = sqrt(eps[j + 1]) * (1 / 2) * (normalized_bessel(n - 1, arg_A) - normalized_bessel(n + 1, arg_A)); // with derivative of Bessel function
            B_m[j][n](1,1) = sqrt(eps[j + 1]) * normalized_hancel_dirivative(n, arg_A); // with derivative of Hankel function

            B_e[j][n](0, 0) = eps[j + 1] * B_m[j][n](0, 0);
            B_e[j][n](0, 1) = eps[j + 1] * B_m[j][n](0, 1);
            B_e[j][n](0, 1) = B_m[j][n](0, 1);
            B_e[j][n](1, 1) = B_m[j][n](1, 1);

            arma::cx_dmat B_m_inv = arma::inv(B_m[j][n]);
            T_m[n] *= B_m_inv * A_m[j][n];

            arma::cx_dmat B_e_inv = arma::inv(B_e[j][n]);
            T_e[n] *= B_e_inv * A_e[j][n];
        }

        /*if (j > 0)
        {
            B_e[j - 1] = A_e[j];
            B_m[j - 1] = A_m[j];
        }*/
    }


    double d_1 = 0;
    double c_1 = 0; 
    d_compl R_e = 1, R_m = 1;
    for (double n = 1; n < N; n ++)
    {
        std::complex<double> c_n = std::pow(i, (n - 1)) / (k * k) * (2 * n + 1) / (n * (n + 1));

        std::complex<double> d_e_n = c_n * (T_e[n](1,0) + T_e[n](1,1) * R_e) / (T_e[n](0,0) + T_e[n](0,1) * R_e);
        std::complex<double> d_m_n = c_n * (T_m[n](1,0) + T_m[n](1,1) * R_m) / (T_m[n](0,0) + T_m[n](0,1) * R_m);

        E_r += d_e_n * (n * (n + 1) / (x * x)) * normalized_hancel(n, k * x) * std::legendre(n, cos(phi)) * cos(phi);
        E_theta = E_theta + (d_e_n * normalized_hancel_dirivative(n,k * x)) + (i * d_m_n / sin(theta) * hancel(n, k * x));
        E_phi = E_phi + (d_e_n / sin(theta) * hancel(n, k * x)) + (i * d_m_n * hancel(n, k * x));
        R_e = d_e_n / c_n;
        R_m = d_m_n / c_n;
        
    }

    E_theta *= k / x * cos(phi);
    E_phi *= -k / x * sin(phi);


}
