

#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include "ndvector.hpp"

// read data from a file into a NDVector
template <int ARITY>
void read_data(const char *filename, NDVector<ARITY> &vec, char sep='\t')
{
    std::ifstream
        file(filename, std::ios::in | std::ios::binary | std::ios::ate);  // open the file at the end
    if (!file.is_open())
    {
        std::cerr << "Error: could not open file " << filename << std::endl;
        exit(1);
    }
    
    size_t size = file.tellg();  // get the size of the file
    file.seekg(0, std::ios::beg);  // go back to the beginning of the file

    // read the file into a buffer
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    file.close();

    // parse the buffer, line by line, tsv format, each col is int separated by a tab
    std::vector<std::array<int, ARITY>> data;
    size_t start = 0;
    size_t end = 0;
    while (end < size)
    {
        std::array<int, ARITY> row;
        for (int i = 0; i < ARITY; i++)
        {
            end = std::find(buffer.begin() + start, buffer.end(), sep) - buffer.begin();
            row[i] = std::stoi(std::string(buffer.begin() + start, buffer.begin() + end));
            start = end + 1;
        }
        data.push_back(row);
    }

    // copy the data into the NDVector
    vec.resize(data.size());
    for (int i = 0; i < data.size(); i++)
    {
        for (int j = 0; j < ARITY; j++)
        {
            vec.vecs[j][i] = data[i][j];
        }
    }
}
