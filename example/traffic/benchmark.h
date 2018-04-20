#ifndef EXAMPLE_TRAFFIC_BENCHMARK_H
#define EXAMPLE_TRAFFIC_BENCHMARK_H

#include <chrono>

template<typename TimeT = std::chrono::microseconds>
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

#endif  // EXAMPLE_TRAFFIC_BENCHMARK_H
