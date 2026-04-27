#ifndef MD_TRIE_DEBUG_H
#define MD_TRIE_DEBUG_H

#include <cstdint>
#include <cstdlib> // for rand()
#include <cstring> // for memset()
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

// BENCHMARKING UTILITIES

#include "profiling_points.h"

// DEBUGGING UTILITIES

// Debug print macro - prints to stderr only if DEBUGF_ENABLED is defined
#ifdef DEBUGF_ENABLED
#define debugf(fmt, ...) fprintf(stderr, fmt, ##__VA_ARGS__)
#else
#define debugf(fmt, ...) ((void)0)
#endif

// Debug the range queries inside each KNN query. 
struct debug_knn_rq { 
    float radius; 
    TimeStamp latency; 
    size_t points_returned; 

    std::string toString() const {
        std::ostringstream oss;
        oss << "KNN_RQ: " << radius << '\t' << TimeStamp_to_us(latency) << '\t'
            << points_returned;
        return oss.str();
    }
};

#endif // MD_TRIE_DEBUG_H
