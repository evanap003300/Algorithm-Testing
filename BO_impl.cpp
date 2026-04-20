#include "BO_impl.hpp"

const int CANIDATE_COUNT = 300;
const int SAMPLE_COUNT = 10;
const int START_COUNT = 5;
const int ITERATIONS = 50;

double BO::normal_cdf(double x)
{
    return 0.5 * (1 + erf(x / sqrt(2)));
}

double BO::normal_pdf(double x)
{
    return exp(-0.5 * x * x) / sqrt(2 * M_PI);
}

double BO::expected_improvement(double mean, double var, double f_best)
{
    double z = (mean - f_best) / sqrt(var);
    double ei = (mean - f_best) * normal_cdf(z) + sqrt(var) * normal_pdf(z);
    return ei;
}

/*
    Creates a grid of candidate x values across the domain
    Computes EI at each one using the fitted GP
    Returns the x with the highest EI.
*/
double BO::find_next_point(BO::Domain domain, double f_best, vector<GP::Point> &sample_data)
{
    auto &[lower_bound, upper_bound] = domain;

    double best_x = lower_bound, best_EI = -1;
    for (int i = 0; i < CANIDATE_COUNT; i++)
    {
        double x = lower_bound + (upper_bound - lower_bound) * i / (CANIDATE_COUNT - 1.0);

        // Skip if too close to existing data
        bool too_close = false;
        for (auto &pt : sample_data)
        {
            if (abs(x - pt.x) < 1e-3)
            {
                too_close = true;
                break;
            }
        }
        if (too_close)
            continue;

        GP::Prediction pred = gp.predict(x);
        double sigma = sqrt(max(pred.variance, 1e-12));
        if (sigma < 1e-10)
            continue;
        double ei = expected_improvement(pred.mean, pred.variance, f_best);
        if (ei > best_EI)
        {
            best_EI = ei;
            best_x = x;
        }
    }
    return best_x;
}

/*
    Initializes with a few random points, then repeatedly fits the GP,
    finds the next point via EI, evaluates the true function, and adds the result to the dataset.
*/
void BO::bo_loop(BO::Domain domain)
{
    auto &[lower_bound, upper_bound] = domain;
    double l = 1;
    double sigma_squared = 9;
    double sigma_n_squared = 0.01;

    // 5 random initial points in the domain
    int initial_points = 5;
    vector<GP::Point> sample_data;
    srand(time(NULL));
    for (int i = 0; i < initial_points; i++)
    {
        double x = lower_bound + (upper_bound - lower_bound) * rand() / (double)RAND_MAX;
        sample_data.push_back({x, f(x)});
    }

    gp = GP(l, sigma_squared, sigma_n_squared, sample_data);
    gp.fit();

    double f_best = -1e9;
    for (auto &point : sample_data)
        f_best = max(f_best, point.y);

    for (int i = 0; i < ITERATIONS; i++)
    {
        double x = find_next_point(domain, f_best, sample_data);
        double y = f(x);
        f_best = max(f_best, y);
        sample_data.push_back({x, y});
        gp = GP(l, sigma_squared, sigma_n_squared, sample_data);
        gp.fit();
        cout << "Iteration " << i + 1 << ": x = " << x
             << ", f(x) = " << y << ", f_best = " << f_best << '\n';
    }
}