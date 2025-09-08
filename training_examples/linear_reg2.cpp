#include <ml/linearRegression.hpp>
#include <ml/utils.hpp>

#include <chrono>
#include <iostream>

int main() {
    auto [data, target_data] = ml::load_csv("data/Admission_Predict.csv", {8});

    /* specify that we want to use gradient descent, with a lr of 0.001 and 5000 iterations */
    ml::linearRegression m(true, 0.001, 5000);

    auto start = std::chrono::high_resolution_clock::now();
    auto training_result = m.train(data, target_data);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> training_time = end - start;
    std::cout << "Training time using Gradient Descent: " << training_time.count() << std::endl;

    /* exclude a factor using the std::vector parameter */
    auto [prediction_data, _] = ml::load_csv("data/Admission_Predict_Test.csv", {8});

    auto predictions = m.predict(training_result, prediction_data);

    ml::write_csv("../output/test_output2.csv", predictions);
}