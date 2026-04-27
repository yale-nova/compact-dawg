#include <cstring>
#include <iostream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <vector>

#ifdef CONFIG_PROFILING_POINTS
#include "profiling_points.h"

// The global profile_point registry (registered in debug.h)
std::vector<profile_point*> profile_point_registry;
std::mutex registry_mutex;

// Returns a JSON array of all global PP data.
nlohmann::json dump_pp_json(void)
{
    std::lock_guard<std::mutex> lock(registry_mutex);

    nlohmann::json pp_array = nlohmann::json::array();

    for (auto *pp : profile_point_registry) {
        pp_array.push_back(pp->toJson());
    }

    return pp_array;
}

// Returns a JSON array as a string.
std::string dump_pps(void) { return dump_pp_json().dump(); }

void print_pps(void) { std::cout << dump_pps() << std::endl; }

void clear_pps(void)
{
    std::lock_guard<std::mutex> lock(registry_mutex);

    for (auto *pp : profile_point_registry) {
        pp->clear();
    }
}

void enable_pps(bool mode)
{
    std::lock_guard<std::mutex> lock(registry_mutex);
    for (auto* pp : profile_point_registry) {
        pp->enable(mode);
    }
}

#else // CONFIG_PROFILING_POINTS

nlohmann::json dump_pp_json() { return nlohmann::json::array(); }
std::string dump_pps() { return "[]"; }
void print_pps() {}
void clear_pps() {}
void enable_pps(bool mode) { (void)mode; }

#endif
