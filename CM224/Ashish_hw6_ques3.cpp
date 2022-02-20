#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#define pi 3.14159265
using namespace std;

int main() {
	double prob_in[2], prob_out[2], mean[2], sigma[2];
	double temp, sum[2];
	sum[0] = 0; sum[1] = 0;
	int iter, data_size;
	prob_in[0] = 0.59;
	prob_in[1] = 0.41;
	mean[0] = 5.55;
	mean[1] = 2.4;
	sigma[0] = 1.03;
	sigma[1] = 0.92;

	ifstream infile("ml_genomics_hw6_data.txt");
	vector<double> data;
	while (infile >> temp) {
		data.push_back(temp);
	}
	data_size = data.size();
	vector<vector<double>> aij(data_size, vector<double>(3, 0));

	for (int i = 0; i < data_size; i++) {
		aij[i][0] = prob_in[0] * prob_in[0] * (exp(-(pow(data[i] - mean[0], 2)) / (2 * pow(sigma[0], 2))) / sigma[0]) * (1 / sqrt(2 * pi));
		aij[i][1] = prob_in[1] * prob_in[1] * (exp(-(pow(data[i] - mean[1], 2)) / (2 * pow(sigma[1], 2))) / sigma[1]) * (1 / sqrt(2 * pi));
		temp = sqrt((pow(sigma[0], 2) + pow(sigma[1], 2)) / 4);
		aij[i][2] = 2*prob_in[0] * prob_in[1] * (exp(-(pow(data[i] - (mean[0]+mean[1])/2, 2)) / (2 * pow(temp, 2))) / temp) * (1 / sqrt(2 * pi));
		temp = aij[i][0] + aij[i][1]+ aij[i][2];
		aij[i][0] /= temp;
		aij[i][1] /= temp;
		aij[i][2] /= temp;
		sum[0] += 2*aij[i][0]+ aij[i][2];
		sum[1] += 2*aij[i][1]+ aij[i][2];
	}
	prob_out[0] = sum[0] / (2 * data_size);
	prob_out[1] = sum[1] / (2 * data_size);
	cout << prob_out[0]<<'\n';
	return 0;
}