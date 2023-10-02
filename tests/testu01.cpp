#include <random>
#include <string>
#include <cstdio>
#include <cstdint>
#include <cstddef>

extern "C" {
    #include "TestU01.h"
}

#include "../include/tyche.h"
#include "../include/squares.h"
#include "../include/phillox.h"

// based on: https://www.pcg-random.org/posts/how-to-test-with-testu01.html


const char* gen_name = "Phillox";  
using RNG = Phillox;

uint32_t gen32()
{
    static RNG rng(42, 0);
    return rng.draw<uint32_t>();
}

// END OF GENERATOR SECTION

inline uint32_t rev32(uint32_t v)
{
    // https://graphics.stanford.edu/~seander/bithacks.html
    // swap odd and even bits
    v = ((v >> 1) & 0x55555555) | ((v & 0x55555555) << 1);
    // swap consecutive pairs
    v = ((v >> 2) & 0x33333333) | ((v & 0x33333333) << 2);
    // swap nibbles ...
    v = ((v >> 4) & 0x0F0F0F0F) | ((v & 0x0F0F0F0F) << 4);
    // swap bytes
    v = ((v >> 8) & 0x00FF00FF) | ((v & 0x00FF00FF) << 8);
    // swap 2-byte-long pairs
    v = ( v >> 16             ) | ( v               << 16);
    return v;
}

uint32_t gen32_rev()
{
    return rev32(gen32());
}


int main ()
{
    // Config options for generator output
    bool reverseBits = false;

    // Name of the generator

    std::string genName = gen_name;

    printf("Testing %s:\n", genName.c_str());
    fflush(stdout);

    // Create a generator for TestU01.

    unif01_Gen* gen =
        unif01_CreateExternGenBits((char*) genName.c_str(),
                                   reverseBits ? gen32 : gen32_rev);

    // Run tests.

    bbattery_SmallCrush(gen);
    fflush(stdout);
    
    // bbattery_Crush(gen);
    // fflush(stdout);
    
    //bbattery_BigCrush(gen);
    //fflush(stdout);

    // Clean up.
    unif01_DeleteExternGenBits(gen);

    return 0;
}