#include <cstdio>
#include <cstdint>

#include "../include/tyche.h"
#include "../include/squares.h"
#include "../include/phillox.h"
#include "../include/threefry.h"


using namespace openrand;


// From build directory, run this as (modify path and generator name as needed):
// ./tests/pract_rand philox | /home/shihab/codes/PractRand/RNG_test stdin32

template <typename RNG>
void generate_stream(){
    for(int ctr = 0; ; ctr++){
        RNG rng(0, ctr);
        for(int i=0; i<10; i++){
            uint32_t value = rng.template draw<uint32_t>();
            fwrite((void*) &value, sizeof(value), 1, stdout);
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Please provide one argument." << std::endl;
        return 1;
    }

    std::string arg = argv[1];

    std::cout << "Generating stream for: " << arg << std::endl;

    if(arg == "philox")
        generate_stream<openrand::Phillox>();
    else if(arg == "tyche")
        generate_stream<openrand::Tyche>();
    else if(arg == "threefry")
        generate_stream<openrand::Threefry>();
    else if(arg == "squares")
        generate_stream<openrand::Squares>();
    else
        std::cout << "Invalid argument." << std::endl;

    return 0;
}