#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <complex>
#include <numbers>
#include <cassert>
#include <armadillo>

#include "Special Functions.h"
#include "Helper Functions.h"

using d_compl = std::complex<double>;

extern std::complex<double> i;
extern double PI;
extern double EPS;


void calculate_electric_field_close(std::vector<double>& r, std::vector<d_compl>& eps_compl, double lambda);

void calculate_electric_field_far(std::vector<double>& r, std::vector<d_compl>& eps_compl, double lambda);
