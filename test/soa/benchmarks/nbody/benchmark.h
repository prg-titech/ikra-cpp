// Taken from: https://stackoverflow.com/questions/2808398/easily-measure-elapsed-time

#ifndef TEST_SOA_BENCHMARK_NBODY_BENCHMARK_H
#define TEST_SOA_BENCHMARK_NBODY_BENCHMARK_H

#include <iostream>
#include <chrono>

template<typename TimeT = std::chrono::milliseconds>
struct measure
{
    template<typename F, typename ...Args>
    static typename TimeT::rep execution(F&& func, Args&&... args)
    {
        auto start = std::chrono::steady_clock::now();
        std::forward<decltype(func)>(func)(std::forward<Args>(args)...);
        auto duration = std::chrono::duration_cast< TimeT> 
                            (std::chrono::steady_clock::now() - start);
        return duration.count();
    }
};

int r_float2int(float value) {
  return *reinterpret_cast<int*>(&value);
}

#endif  // TEST_SOA_BENCHMARK_NBODY_BENCHMARK_H
