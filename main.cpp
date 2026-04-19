#include "GP_impl.hpp"
#include <cmath>
#include <iostream>
#include <vector>

using namespace std;

int main()
{
    double l = 1.0;
    double sigma_squared = 1.0;
    double sigma_n_squared = 0.1;
    vector<GP::Point> sample_data = {
        {0.1, sin(2 * M_PI * 0.1)},
        {0.3, sin(2 * M_PI * 0.3)},
        {0.5, sin(2 * M_PI * 0.5)},
        {0.7, sin(2 * M_PI * 0.7)},
        {0.9, sin(2 * M_PI * 0.9)}};

    GP gp(l, sigma_squared, sigma_n_squared, sample_data);

    gp.fit();

    for (int i = 0; i <= 200; i++)
    {
        double x_star = i / 200.0;
        GP::Prediction pred = gp.predict(x_star);
        cout << x_star << "," << pred.mean << "," << pred.variance << endl;
    }
    return 0;
}