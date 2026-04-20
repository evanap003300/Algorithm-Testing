#include "GP_impl.hpp"
#include "BO_impl.hpp"
#include <cmath>
#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

int main()
{
    BO bo;
    BO::Domain domain = {-20.0, 20.0};
    bo.bo_loop(domain);
    return 0;
}