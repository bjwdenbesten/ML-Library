#pragma once

#include <vector>

namespace ml {
    class linearRegression {
        public:
            linearRegression(bool use_gd, double lr, int it) {
                learning_rate = lr;
                iterations = it;
                useGradientDescent = use_gd;
            }
            bool useGradientDescent;
            double learning_rate;
            int iterations;
            std::vector<std::vector<double>> train(const std::vector<std::vector<double>> &data, const std::vector<std::vector<double>> &target_data);
            std::vector<std::vector<double>> predict(const std::vector<double> &b_vals, const std::vector<std::vector<double>> &ind_data);
        private:
            std::vector<std::vector<double>> lin_reg_normaleq(const std::vector<std::vector<double>> &data, const std::vector<std::vector<double>> &target);
            std::vector<std::vector<double>> lin_reg_gd(const std::vector<std::vector<double>> &data, const std::vector<std::vector<double>> &target);
    };
}