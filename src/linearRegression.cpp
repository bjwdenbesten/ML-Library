#include "ml/linearRegression.hpp"

#include <Eigen/Dense>
#include <iostream>

namespace ml {
    std::vector<std::vector<double>> linearRegression::train(std::vector<std::vector<double>> &data, const std::vector<std::vector<double>> &target_data) {
        if (useGradientDescent) {
            return lin_reg_gd(data, target_data);
        }
        else {
            return lin_reg_normaleq(data, target_data);
        }
    }

    std::vector<std::vector<double>> linearRegression::predict(const std::vector<std::vector<double>> &b_vals, std::vector<std::vector<double>> &ind_data) {
        int samples = ind_data.size();
        int features = ind_data[0].size();
        int outputs = b_vals[0].size();

        normalizeFromTraining(ind_data);

        std::vector<std::vector<double>> predictions(samples, std::vector<double>(outputs));

        Eigen::MatrixXd X(samples, features + 1);
        X.col(0) = Eigen::VectorXd::Ones(samples);

        for (int j = 0; j < features; j++) {
            Eigen::VectorXd col(samples);
            for (int i = 0; i < samples; i++) {
                col(i) = ind_data[i][j];
            }
            X.col(j + 1) = col;
        }

        Eigen::MatrixXd B(b_vals.size(), b_vals[0].size());
        for (int i = 0; i < b_vals.size(); i++) {
            for (int j = 0; j < b_vals[0].size(); j++) {
                B(i, j) = b_vals[i][j];
            }
        }

        Eigen::MatrixXd preds = X * B;

        for (int i = 0; i < preds.rows(); i++) {
            for (int j = 0; j < preds.cols(); j++) {
                predictions[i][j] = preds(i, j);
            }
        }
        return predictions;
    }

    void linearRegression::normalize(std::vector<std::vector<double>> &data) {
        int features = data[0].size();
        int data_size = data.size();

        /* clear cached min / maxes in class */
        min_vals.clear();
        max_vals.clear();

        for (size_t i = 0; i < features; i++) {
            double mn = data[0][i];
            double mx = data[0][i];

            for (size_t j = 0; j < data_size; j++) {
                if (data[j][i] < mn) mn = data[j][i];
                if (data[j][i] > mx) mx = data[j][i];
            }
            min_vals.push_back(mn);
            max_vals.push_back(mx);

            if (mn != mx) {
                for (size_t k = 0; k < data_size; k++) {
                    data[k][i] = (data[k][i] - mn) / (mx - mn);
                }
            }
        }
    }

    std::vector<std::vector<double>> linearRegression::lin_reg_normaleq(std::vector<std::vector<double>> &data, const std::vector<std::vector<double>> &target) {
        int n = data.size();
        int m = data[0].size();
        Eigen::MatrixXd X(n, m + 1);

        normalize(data);

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

        Eigen::MatrixXd beta_vals = X.colPivHouseholderQr().solve(Y);
        std::vector<std::vector<double>> res(beta_vals.rows(), std::vector<double>(beta_vals.cols()));
        for (int i = 0; i < beta_vals.rows(); i++) {
            for (int j = 0; j < beta_vals.cols(); j++) {
                res[i][j] = beta_vals(i, j);
            }
        }
        return res;
    }

    std::vector<std::vector<double>> linearRegression::lin_reg_gd(std::vector<std::vector<double>> &data, const std::vector<std::vector<double>> &target) {
        /* normalize to avoid nan values */
        normalize(data);
        
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

        int dep_variables = target[0].size();

        Eigen::MatrixXd weights = Eigen::MatrixXd::Random(m + 1, dep_variables) * 0.01;

        double alpha = learning_rate;
        double dec = 0.001;

        for (int i = 0; i < iterations; i++) {
            Eigen::MatrixXd err = X * weights - Y;
            Eigen::MatrixXd gradient = (X.transpose() * err) / n;

            double alpha_transform = alpha * std::exp(-dec * i);

            weights = weights - alpha_transform * gradient;
        }

        std::vector<std::vector<double>> res(weights.rows(), std::vector<double>(weights.cols()));

        for (int i = 0; i < weights.rows(); i++) {
            for (int j= 0; j < weights.cols(); j++) {
                res[i][j] = weights(i, j);
            }
        }
        return res;
    }

    void linearRegression::normalizeFromTraining(std::vector<std::vector<double>> &data) {
        int features = data[0].size();
        int data_size = data.size();

        if (min_vals.size() == 0 || max_vals.size() == 0) {
            throw std::runtime_error("No normalization data avaliable");
        }

        for (size_t i = 0; i < features; i++) {
            if (min_vals[i] != max_vals[i]) {
                for (size_t k = 0; k < data_size; k++) {
                    data[k][i] = (data[k][i] - min_vals[i]) / (max_vals[i] - min_vals[i]);
                }
            }
        }
    }
}



