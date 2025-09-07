#include "ml/utils.hpp"
#include "ml/linearRegression.hpp"

#include <iostream>


int main() {
    auto [data, col_data] = ml::load_csv("data/test_input.csv", {2});
    ml::write_csv("../output/test_file.csv", col_data);
}