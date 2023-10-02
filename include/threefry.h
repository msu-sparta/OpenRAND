#ifndef THREEFRY_H
#define THREEFRY_H

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>

#include "base_state.hpp"


class Threefry: public BaseRNG<Threefry>{
public:

    DEVICE Threefry(uint64_t seed, uint32_t ctr, uint32_t global_seed=rnd::DEFAULT_GLOBAL_SEED)
    : seed(seed ^ global_seed),
      counter(ctr)
    {

    }

    template <typename T = uint32_t> DEVICE T draw() {
        uint32_t out[2];
        round(static_cast<uint32_t>(seed >> 32),
                                static_cast<uint32_t>(seed),
                                counter,
                                _ctr++,
                                out);

        static_assert(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>);
        if constexpr (std::is_same_v<T, uint32_t>)
            return out[0];

        uint64_t res =
            (static_cast<uint64_t>(out[0]) << 32) | static_cast<uint64_t>(out[1]);
        return static_cast<uint64_t>(res);
    }

private:

  DEVICE uint32_t rotl32(uint32_t x, uint32_t N){
      return (x << (N & 31)) | (x >> ((32-N) & 31));
  }


  DEVICE void round(uint32_t ks0, uint32_t ks1, uint32_t counter, uint32_t _ctr, uint32_t* out){
        uint32_t x0, x1;
        uint32_t ks2 = 0x1BD11BDA;
                                                    
                                                      
        x0  = counter + ks0;                                                  
        ks2 ^= ks0; 

        x1  = _ctr + ks1;                                                  
        ks2 ^= ks1;

        for(int i=0; i<20; i++){
            x0 += x1;
            x1 = rotl32(x1, get_constant(i%8));
            x1 ^= x0;

            if(i==3){
                x0 += ks1;
                x1 += ks2;
                x1 += 1;
            }
            if(i==7){
                x0 += ks2;
                x1 += ks0;
                x1 += 2;
            }
            if(i==11){
                x0 += ks0;
                x1 += ks1;
                x1 += 3;
            }
            if(i==15){
                x0 += ks1;
                x1 += ks2;
                x1 += 4;
            }
            if(i==19){
                x0 += ks2;
                x1 += ks0;
                x1 += 5;
            }
        } 
        out[0] = x0;
        out[1] = x1;
    }


    DEVICE int get_constant(const int index) const {
        switch (index) {
            case 0: return 13;
            case 1: return 15;
            case 2: return 26;
            case 3: return 6;
            case 4: return 17;
            case 5: return 29;
            case 6: return 16;
            case 7: return 24;
            default: return 24;
        }
    }

    const uint64_t seed;
    const uint32_t counter;
    uint32_t _ctr = 0;

};

#endif // THREEFRY_H