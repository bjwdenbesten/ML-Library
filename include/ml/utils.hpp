#pragma once

#include <vector>
#include <string>

namespace ml {
    /* function to read a csv with a target column*/
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
    load_csv(const std::string &file_name, std::vector<int> target_cols);
    
    /* function to write target/prediction data to a csv file */
    void write_csv(const std::string &file_name, std::vector<std::vector<double>> &data);
}