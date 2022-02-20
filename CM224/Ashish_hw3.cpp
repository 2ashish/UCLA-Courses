// CM224 Homework 3
// Author: Ashish Kumar Singh
// UID: 105479019

#include<iostream>
using namespace std;

double func(double x, double y) {
	return (9 - 6 * x + x * x + 100 * y * y + 200 * x * x * y + 100 * x * x * x * x);
}
double func_x(double x, double y) {
	return (-6 + 2 * x + 400 * x * y + 400 * x * x * x);
}
double func_y(double x, double y) {
	return (200 * y + 200 * x * x);
}

int main() {
	double x_prev, y_prev, prev,cur,x_cur,y_cur,stop_cri;
	double eps[3] = { 1e-3 ,1e-4,1e-5};
	stop_cri = 1e-8;
	for (int eps_i = 0; eps_i < 3; eps_i++) {
		x_prev = 2; y_prev = 2;
		prev = func(x_prev, y_prev);
		long int iter = 0;
		for (iter = 1; iter < 1e7; iter++) {
			x_cur = x_prev - eps[eps_i] * func_x(x_prev, y_prev);
			y_cur = y_prev - eps[eps_i] * func_y(x_prev, y_prev);
			cur = func(x_cur, y_cur);
			if (abs(prev - cur) < stop_cri)break;
			//if (iter % 100 == 0)cout << iter << " " << cur << '\n';
			prev = cur;
			x_prev = x_cur;
			y_prev = y_cur;
		}
		cout << "For eps of: "<<eps[eps_i]<<" last iter = "<<iter << " f(x,y) = " << cur <<" x = "<<x_cur<<" y = "<<y_cur<<'\n';
	}
	
	
	return 0;
}