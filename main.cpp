#include "Sphere difraction.h"

void FirstExercise()
{
	const double lambda = 2 * PI / 1000.0; // wave length
	const double k = 2 * PI / lambda; // wave number
	int layers_number = 3;

	std::vector<d_compl> eps = { 1, d_compl(1.1, 0.01), 2.5 }; // permittivities of layers
	eps.push_back(8.85e-12); // permittivity of vacuum (outer space)

	std::vector<double> r = { 
		0.01, 
		0.01 + PI / k, 
		0.01 + PI / k + 0.5 / k 
	}; // concentric spheres radiuses

	calculate_electric_field_far(r, eps, lambda);
}

void SimpleSphere()
{
	double lambda = 2 * PI / 1000.0; // wave length
	double k = 2 * PI / lambda; // wave number
	int layers_number = 1;

	std::vector<d_compl> eps = { 1, 8.85e-12 };

	std::vector<double> r = { 0.01 };

	calculate_electric_field_far(r, eps, lambda);
}

int main()
{
	FirstExercise();
	//SimpleSphere();
}
