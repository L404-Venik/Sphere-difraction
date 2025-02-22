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

    if constexpr (std::is_same_v<N, std::complex<double>> || std::is_same_v<N, std::complex<float>>)
    {
        for (const auto& val : v)
        {
            double realPart = val.real();
            double imagPart = val.imag();

            if (std::abs(realPart) < 1e-5 && realPart != 0)
                file << std::scientific << std::setprecision(precision) << realPart;
            else
                file << std::fixed << std::setprecision(precision) << realPart;

            if (std::abs(imagPart) < 1e-5 && imagPart != 0)
                file << " " << std::scientific << std::setprecision(precision) << imagPart << "i\n";
            else
                file << " " << std::fixed << std::setprecision(precision) << imagPart << "i\n";
        }
    }
    else
    {
        for (const auto& val : v)
        {
            if (std::abs(val) < 1e-5 && val != 0)
                file << std::scientific << std::setprecision(precision) << val << "\n";
            else
                file << std::fixed << std::setprecision(precision) << val << "\n";
        }
    }

	file.close();
}

void writeComplexVectorsToFile(const std::string& filename,
	const std::vector<std::complex<double>>& v1,
	const std::vector<std::complex<double>>& v2,
	const std::vector<std::complex<double>>& v3,
	int precision = 6);