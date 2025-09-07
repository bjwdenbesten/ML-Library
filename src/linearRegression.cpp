#include "ml/linearRegression.hpp"

#include <Eigen/Dense>

namespace ml {
    std::vector<std::vector<double>> linearRegression::train(const std::vector<std::vector<double>> &data, const std::vector<std::vector<double>> &target_data) {
        if (useGradientDescent) {
            return lin_reg_gd(data, target_data);
        }
        else {
            return lin_reg_normaleq(data, target_data);
        }
    }

    std::vector<std::vector<double>> linearRegression::predict(const std::vector<double> &b_vals, const std::vector<std::vector<double>> &ind_data) {
        int n = ind_data.size();
        int m = ind_data[0].size();
        std::vector<std::vector<double>> predictions(n);

        /* finish this */

        return predictions;
    }

    std::vector<std::vector<double>> lin_reg_normaleq(const std::vector<std::vector<double>> &data, const std::vector<std::vector<double>> &target) {
        int n = data.size();
        int m = data[0].size();
        Eigen::MatrixXd X(n, m + 1);
        for (int i = 0; i < n; i++) {
            X(i, 0) = 1.0;
            for (int j = 1; j < m + 1; j++) {
                X(i, j) = data[i][j-1];
            }
        }
        Eigen::MatrixXd Y(n, 1);
        for (int i = 0; i < target.size(); i++) {
            for (int j = 0; j < target[0].size(); j++) {
                Y(i, j) = target[i][j];
            }
        }

        Eigen::MatrixXd beta_vals = ((X.transpose() * X).inverse() * (X.transpose() * Y));
        std::vector<std::vector<double>> res(beta_vals.size());
        for (int i = 0; i < beta_vals.rows(); i++) {
            for (int j = 0; j < beta_vals.cols(); j++) {
                res[i][j] = beta_vals(i, j);
            }
        }
        return res;
    }

    std::vector<std::vector<double>> lin_reg_gd(const std::vector<std::vector<double>> &data, const std::vector<std::vector<double>> &target) {
        
    }
}



