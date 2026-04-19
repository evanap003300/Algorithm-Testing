#include "GP_impl.hpp"

double GP::kernel(double x1, double x2)
{
    double diff = x1 - x2;
    return sigma_squared * exp(-0.5 * (diff * diff) / (l * l));
}

vector<vector<double>> GP::kernel_matrix()
{
    int n = sample_data.size();
    vector<vector<double>> K(n, vector<double>(n, 0.0));

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            double x1 = sample_data[i].x, x2 = sample_data[j].x;
            K[i][j] = kernel(x1, x2) + delta(i, j) * sigma_n_squared;
            K[j][i] = K[i][j]; // symmetric
        }
    }

    return K;
}

// K = L * L^T
vector<vector<double>> GP::cholesky_decomp(vector<vector<double>> &K)
{
    int n = K.size();
    vector<vector<double>> L(n, vector<double>(n, 0.0));
    for (int j = 0; j < n; j++)
    {
        double sum = 0;
        for (int k = 0; k < j; k++)
            sum += L[j][k] * L[j][k];
        L[j][j] = sqrt(K[j][j] - sum);

        for (int i = j + 1; i < n; i++)
        {
            double sum = 0;
            for (int k = 0; k < j; k++)
                sum += L[i][k] * L[j][k];
            L[i][j] = (K[i][j] - sum) / L[j][j];
        }
    }

    return L;
}

// L * Z = Y
vector<double> GP::forward_substitution(vector<vector<double>> &L, vector<double> &Y)
{
    int n = L.size();
    vector<double> Z(n);
    for (int i = 0; i < n; i++)
    {
        double sum = 0;
        for (int j = 0; j < i; j++)
        {
            sum += L[i][j] * Z[j];
        }
        Z[i] = (Y[i] - sum) / L[i][i];
    }
    return Z;
}

// L^T * W = Z
vector<double> GP::backward_substitution(vector<vector<double>> &L, vector<double> &Z)
{
    int n = L.size();
    vector<double> W(n);

    for (int i = n - 1; i >= 0; i--)
    {
        double sum = 0;
        for (int j = i + 1; j < n; j++)
        {
            sum += L[j][i] * W[j];
        }
        W[i] = (Z[i] - sum) / L[i][i];
    }

    return W;
}

GP::Prediction GP::predict(double x_star, vector<double> &W, vector<vector<double>> &L)
{
    int n = sample_data.size();
    vector<double> k_star(n);

    for (int i = 0; i < n; i++)
    {
        k_star[i] = kernel(x_star, sample_data[i].x);
    }

    double mean = 0;
    for (int i = 0; i < n; i++)
    {
        mean += k_star[i] * W[i];
    }

    vector<double> v = forward_substitution(L, k_star);
    double v_dot_v = 0, x_star_kernel = kernel(x_star, x_star);
    for (int i = 0; i < v.size(); i++)
    {
        v_dot_v += v[i] * v[i];
    }

    double variance = x_star_kernel - v_dot_v; // (prior variance) - (reduction in uncertainty)
    return {mean, variance};
}