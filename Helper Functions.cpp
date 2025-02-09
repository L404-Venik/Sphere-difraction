#include "Helper Functions.h"

void writeComplexVectorsToFile(const std::string& filename,
	const std::vector<std::complex<double>>& v1,
	const std::vector<std::complex<double>>& v2,
	const std::vector<std::complex<double>>& v3,
	int precision) 
{
	std::ofstream file(filename);
	if (!file) {
		std::cerr << "Ошибка открытия файла!" << std::endl;
		return;
	}

	file << std::fixed << std::setprecision(precision);
	size_t maxSize = std::max({ v1.size(), v2.size(), v3.size() });

	for (size_t i = 0; i < maxSize; ++i) {
		if (i < v1.size()) file << v1[i].real() << "+" << v1[i].imag() << "i";
		file << "\t";
		if (i < v2.size()) file << v2[i].real() << "+" << v2[i].imag() << "i";
		file << "\t";
		if (i < v3.size()) file << v3[i].real() << "+" << v3[i].imag() << "i";
		file << "\n";
	}

	file.close();
}