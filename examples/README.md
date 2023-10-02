## Examples



## Statictical Tests

+ `testu01.cpp`
Runs the TestU01 statistical test suite on a specified generator. The path to the TestU01 library 
has to be defined while configuring the project with CMake.

+ `testu01-multistream.cpp`
Checks for correlations between streams when multiple generators are used in parallel using 
the TestU01. The path to the TestU01 library has to be defined.

+ `pract_rand.cpp`
You need to have PractRand installed on your system. To test using PractRand, pipe the output of 
the executable `examples/pract_rand` to the `RNG_test` executable in the library. For example- \
```./examples/pract_rand | ./RNG_test stdin32```

Note: the most stringent versions of these tests can take upto days to complete. Defaults chosen
here should complete within an hour or less. 