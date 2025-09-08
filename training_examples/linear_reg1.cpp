#include "ml/utils.hpp"
#include "ml/linearRegression.hpp"

#include <iostream>
#include <chrono>


int main() {
    /* load csv file into two 2-D matrices */
    auto [data, col_data] = ml::load_csv("data/Admission_Predict.csv", {8});

    /* create instance of the default linearRegression class (using normalEQ)*/
    ml::linearRegression m;
    
    /* train the data, and benchmark */
    auto start = std::chrono::high_resolution_clock::now();
    auto res_train = m.train(data, col_data);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> training_time = end - start;
    std::cout << "Time to train with Normal EQ: " << training_time.count() << std::endl;

    /* now load csv for data to predict */
    auto [predict_data, _] = ml::load_csv("data/Admission_Predict_Test.csv", {8});

    /* predict using data, and training result */
    auto res_predict = m.predict(res_train, predict_data);

    /* write predictions to a csv*/
    ml::write_csv("../output/test_output1.csv", res_predict);
}