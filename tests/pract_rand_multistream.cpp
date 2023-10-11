#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <string>

#include "../include/tyche.h"
#include "../include/squares.h"
#include "../include/phillox.h"
#include "../include/threefry.h"


using namespace openrand;

// Control parameters for this test program.
const int C = 3; // C numbers from per stream
const int NS = 100; // from NS streams
// end


template <typename RNG>
void init_generators(std::vector<RNG> &generators, const int ctr){
    for(int i = 0; i < NS; i++){
        generators.emplace_back(i, ctr);
    }
}

template <typename RNG>
void populate_buffer(std::vector<RNG> &generators, std::vector<uint32_t> &buffer){
    for(int i = 0; i < NS; i++){
        for(int j = 0; j < C; j++){
            buffer.push_back(generators[i].template draw<uint32_t>());
        }
    }
    std::reverse(buffer.begin(), buffer.end());
}

template <typename RNG>
void generate_multi_stream(){
    std::vector<RNG> generators;
    std::vector<uint32_t> buffer;

    for(int ctr = 0; ; ctr++){
        init_generators(generators, ctr);
        populate_buffer(generators, buffer);
        fwrite((void*) buffer.data(), sizeof(buffer[0]), NS * C, stdout);
        
        // clear both vectors
        buffer.clear();
        generators.clear();
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Please provide one argument." << std::endl;
        return 1;
    }

    std::string arg = argv[1];

    std::cout << "Generating multi-stream for: " << arg << std::endl;

    if(arg == "philox")
        generate_multi_stream<Phillox>();
    else if(arg == "tyche")
        generate_multi_stream<Tyche>();
    else if(arg == "threefry")
        generate_multi_stream<Threefry>();
    else if(arg == "squares")
        generate_multi_stream<Squares>();
    else
        std::cout << "Invalid argument." << std::endl;

    return 0;
}