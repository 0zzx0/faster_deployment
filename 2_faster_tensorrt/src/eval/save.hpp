#pragma once

#include <fstream>
#include <string>
#include <vector>

class SaveResult {
public:
    SaveResult(std::string &filename) {
        out.open(filename);
    }
    ~SaveResult() {
        if(out.is_open()) {
            out.close();
        }
    }

    void save_one_line(std::string &img_name, std::string & image_id, int category_id, float score, std::vector<int> &result) {
        if(out.is_open()) {
            out << img_name << " "<< image_id << " "<< category_id << " " << score << " ";
            for(auto &i : result) {
                out << i << " ";
            }
            out << "\n";
        }
    }

private:
    std::ofstream out;
};


