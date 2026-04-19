#pragma once
#include <stdint.h>
#include <vector>

using namespace std;

class GP
{
public:
    struct Point
    {
        double x;
        double y;
    };

    struct Prediction
    {
        double mean;
        double variance;
    };

    GP(double l, double sigma_squared, double sigma_n_squared, vector<Point> sample_data)
        : l(l), sigma_squared(sigma_squared), sigma_n_squared(sigma_n_squared), sample_data(sample_data) {}

    vector<vector<double>> kernel_matrix();
    vector<vector<double>> cholesky_decomp(vector<vector<double>> &K);
    vector<double> forward_substitution(vector<vector<double>> &L, vector<double> &Y);
    vector<double> backward_substitution(vector<vector<double>> &L, vector<double> &Z);

    void fit()
    {
        auto K = kernel_matrix();
        L = cholesky_decomp(K);
        vector<double> y(sample_data.size());
        for (size_t i = 0; i < sample_data.size(); i++)
            y[i] = sample_data[i].y;
        auto z = forward_substitution(L, y);
        W = backward_substitution(L, z);
    }

    Prediction predict(double x_star, vector<double> &W, vector<vector<double>> &L);
    Prediction predict(double x_star)
    {
        return predict(x_star, W, L);
    }

private:
    double kernel(double x1, double x2);
    double delta(int i, int j) { return i == j ? 1.0 : 0.0; }

    vector<vector<double>> L;
    vector<double> W;

    double l;
    double sigma_squared;
    double sigma_n_squared; // noise variance
    vector<Point> sample_data;
};