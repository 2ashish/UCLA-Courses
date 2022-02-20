#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#define pi 3.14159265
using namespace std;

int main() {
	double prob[2][2], mean[2][2], sigma[2][2];
	double temp, sum[6];
	int iter,data_size;
	prob[0][0] = 0.6;
	prob[1][0] = 0.4;
	mean[0][0] = 5.6;
	mean[1][0] = 4.5;
	sigma[0][0] = 1.05;
	sigma[1][0] = 0.95;

	ifstream infile("ml_genomics_hw6_data.txt");
	vector<double> data;
	while (infile >> temp) {
		data.push_back(temp);
	}
	data_size = data.size();

	vector<vector<double>> aij(data_size, vector<double>(2,0));
	
	
	for (iter = 1; iter < 1e5; iter++) {
		//aij calculation
		sum[0] = 0; sum[1] = 0; sum[2] = 0; sum[3] = 0; sum[4] = 0; sum[5] = 0;
		for (int i = 0; i < data_size; i++) {
			aij[i][0] = prob[0][0] * (exp(-(pow(data[i] - mean[0][0], 2)) / (2 * pow(sigma[0][0], 2))) / sigma[0][0]) * (1 / sqrt(2 * pi));
			aij[i][1] = prob[1][0] * (exp(-(pow(data[i] - mean[1][0], 2)) / (2 * pow(sigma[1][0], 2))) / sigma[1][0]) * (1 / sqrt(2 * pi));
			temp = aij[i][0] + aij[i][1];
			aij[i][0] /= temp;
			aij[i][1] /= temp;
			sum[0] += aij[i][0];
			sum[1] += aij[i][1];
			sum[2] += aij[i][0] * data[i];
			sum[3] += aij[i][1] * data[i];
			sum[4] += aij[i][0] * pow(data[i] - mean[0][0], 2);
			sum[5] += aij[i][1] * pow(data[i] - mean[1][0], 2);
		}
		//new prob calulcation
		prob[0][1] = sum[0] / data_size;
		prob[1][1] = sum[1] / data_size;

		//new mean calulcation
		mean[0][1] = sum[2] / sum[0];
		mean[1][1] = sum[3] / sum[1];

		//new sigma calulcation
		sigma[0][1] = sqrt(sum[4] / sum[0]);
		sigma[1][1] = sqrt(sum[5] / sum[1]);

		if (abs(prob[0][1] - prob[0][0]) <= 1e-6)break;

		//update value
		prob[0][0] = prob[0][1];
		prob[1][0] = prob[1][1];
		mean[0][0] = mean[0][1];
		mean[1][0] = mean[1][1];
		sigma[0][0] = sigma[0][1];
		sigma[1][0] = sigma[1][1];

	}
	cout << "converged at iter: " << iter << '\n';
	cout << "pop1 prob : " << prob[0][0] << " mean: " << mean[0][0] << " sigma: " << sigma[0][0] << '\n';
	cout << "pop2 prob : " << prob[1][0] << " mean: " << mean[1][0] << " sigma: " << sigma[1][0] << '\n';
	return 0;
}
