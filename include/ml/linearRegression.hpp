#pragma once

#include <vector>

namespace ml {
    class linearRegression {
        public:
            linearRegression(bool use_gd = false, double lr = 0.0001, int it = 1000) {
                learning_rate = lr;
                iterations = it;
                useGradientDescent = use_gd;
            }
            bool useGradientDescent;
            double learning_rate;
            int iterations;
            std::vector<std::vector<double>> train(std::vector<std::vector<double>> &data, const std::vector<std::vector<double>> &target_data);
            std::vector<std::vector<double>> predict(const std::vector<std::vector<double>> &b_vals, std::vector<std::vector<double>> &ind_data);
            void normalize(std::vector<std::vector<double>> &data);
        private:
            std::vector<double> min_vals;
            std::vector<double> max_vals;
            void normalizeFromTraining(std::vector<std::vector<double>> &data);

            std::vector<std::vector<double>> lin_reg_normaleq(std::vector<std::vector<double>> &data, const std::vector<std::vector<double>> &target);
            std::vector<std::vector<double>> lin_reg_gd(std::vector<std::vector<double>> &data, const std::vector<std::vector<double>> &target);
    };
}