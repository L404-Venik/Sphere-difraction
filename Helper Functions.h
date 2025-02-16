#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <limits>
#include <complex>
#include <iomanip>


template<typename N>
void writeNumericVectorToFile(const std::string& filename, const std::vector<N>& v, int precision = 6)
{
	std::ofstream file(filename);
	if (!file)
	{
		std::cerr << "Ошибка открытия файла!" << std::endl;
		return;
	}

	file << std::fixed << std::setprecision(precision);

	for (size_t i = 0; i < v.size(); ++i)
		file << v[i] << "\n";

	file.close();
}

void writeComplexVectorsToFile(const std::string& filename,
	const std::vector<std::complex<double>>& v1,
	const std::vector<std::complex<double>>& v2,
	const std::vector<std::complex<double>>& v3,
	int precision = 6);