#pragma once
#include <cmath>
#include <vector>
#include "GP_impl.hpp"
#include <iostream>

using namespace std;

class BO
{
public:
    struct Domain
    {
        double lower_bound;
        double upper_bound;
    };

    double normal_cdf(double x);
    double normal_pdf(double x);
    double expected_improvement(double mean, double var, double f_best);
    double f(double x) { return sin(0.2 * M_PI * x) + 7 * cos(x) + 3; };
    double find_next_point(Domain domain, double f_best, vector<GP::Point> &sample_data);
    void bo_loop(BO::Domain domain);
    GP gp;
};