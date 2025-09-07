#include "ml/utils.hpp"

#include <fstream>
#include <ostream>
#include <sstream>
#include <iostream>
#include <map>

namespace ml {
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
    load_csv(const std::string &file_name, std::vector<int> target_cols) {
        std::fstream in_file(file_name, std::ios::in);
        if (!in_file.is_open()) {
            throw std::runtime_error("Could not open csv file.");
        }

        bool has_target = true;
        if (target_cols.empty()) has_target = false;

        std::map<int, bool> target_map;
        if (has_target) {
            for (int i = 0; i < target_cols.size(); i++) {
                int target = target_cols[i];
                target_map[target] = true;
            }
        }

        std::vector<std::vector<double>> data{};
        std::vector<std::vector<double>> target_col_data{};
        std::string line;
        int row_number = 0;

        while (std::getline(in_file, line)) {
            std::stringstream line_stream(line);
            std::vector<double> row{};
            std::vector<double> target_row{};
            std::string value;
            double d_value;
            int curr_col = 0;
            bool flag = false;
            
            while (std::getline(line_stream, value, ',')) {
                try {
                    d_value = std::stod(value);
                }
                catch (...) {
                    curr_col++;
                    flag = true;
                    continue;
                }

                if (has_target && target_map[curr_col]) {
                    target_row.push_back(d_value);
                }
                else {
                    row.push_back(d_value);
                }
                curr_col++;
            }
            if (!row.empty()) {
                data.push_back(row);
            }
            if (!flag) {
                target_col_data.push_back(target_row);
            }
        }
        return {data, target_col_data};
    }

    void write_csv(const std::string &file_name, std::vector<std::vector<double>> &data) {
        std::ofstream out_file(file_name);
        if (!out_file.is_open()) {
            throw std::runtime_error("Couldn't open output file.");
        }



        for (size_t i = 0; i < data.size(); i++) {
            std::vector<double> &row = data[i];
            for (size_t j = 0; j < row.size(); j++) {
                out_file << row[j];
                if (j != row.size() - 1) out_file << ',';
            }
            if (i != data.size() - 1) out_file << std::endl;
        }
    }
}
