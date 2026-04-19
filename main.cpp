#include "GP_impl.hpp"
#include <cmath>
#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

int main()
{
    double l = 0.3;
    double sigma_squared = 9.0;
    double sigma_n_squared = 0.1;
    vector<GP::Point> sample_data = {
        {0.1, sin(2 * M_PI * 0.1)},
        {0.3, sin(2 * M_PI * 0.3)},
        {0.5, sin(2 * M_PI * 0.5)},
        {0.7, sin(2 * M_PI * 0.7)},
        {0.9, sin(2 * M_PI * 0.9)}};

    GP gp(l, sigma_squared, sigma_n_squared, sample_data);

    gp.fit();

    ofstream out("gp_output.csv");
    out << "x,mean,variance,lower,upper" << endl;

    for (int i = 0; i <= 200; i++)
    {
        double x_star = i / 200.0;
        auto pred = gp.predict(x_star);
        double sigma = sqrt(max(pred.variance, 0.0));
        out << x_star << ","
            << pred.mean << ","
            << pred.variance << ","
            << pred.mean - 2 * sigma << ","
            << pred.mean + 2 * sigma << endl;
    }
    out.close();
    return 0;
}