#ifndef PROFILING_POINTS_H
#define PROFILING_POINTS_H

#include <cstdint>
#include <cstdlib> // for rand()
#include <cstring> // for memset()
#include <iomanip>
#include <mutex>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <vector>
#include <x86intrin.h>

// -----------------------------------------------------
// BENCHMARKING UTILITIES
// -----------------------------------------------------

// TODO: put this in a proper namespace for later.
//       I think we broke all of the prior benchmarks which used the old GetTimeStamp.
typedef unsigned long long int TimeStamp;

// Try to avoid using this function directly. Instead, call
// PP_SAVE_TIMESTAMP(), which automatically disables itself when
// profiling is disabled. See its docs for more info!
[[maybe_unused]] static inline TimeStamp GetTimestamp()
{
    unsigned int aux;
    return __rdtscp(&aux);
}

[[maybe_unused]] static double TimeStamp_to_us(TimeStamp ts)
{
    static_assert(sizeof(uint64_t) == sizeof(TimeStamp));

    // On VMHost7: the RDTSC counter increments at 2400.005 MHz
    // (as determined by the kernel during boot-up). It is unaffected by
    // CPU frequency scaling. Please change this function when running
    // benchmarks on a different server!
    unsigned long long int timer_ticks = ts;

    // 2.4e9 ticks per second => 2400 ticks per microsecond
    return ((double)timer_ticks / 2400.0);
}

[[maybe_unused]] static uint64_t TimeStamp_to_cycles(TimeStamp ts)
{
    static_assert(sizeof(uint64_t) == sizeof(TimeStamp));
    return ts;
}

// -----------------------------------------------------
// PROFILING POINTS: Copied from Mage Project
// -----------------------------------------------------

// to string
std::string dump_pps();
// to json array (for combining with other PPs)
nlohmann::json dump_pp_json();
// print them!
void print_pps();
// reset them!
void clear_pps();
// toggle them! (sorry for confusing name)
void enable_pps(bool mode);

#ifdef CONFIG_PROFILING_POINTS
#define PP_PROFILING_ENABLED true

#define PP_SAMPLE_MIN_DELAY 8
#define PP_SAMPLE_MAX_DELAY 64
#define PP_SAMPLE_BUF_SIZE 128
#define PP_MAX_NAME_LEN 64

// Forward declaration
struct profile_point;

// Global registry for profile points
extern std::vector<profile_point *> profile_point_registry;
extern std::mutex registry_mutex;

// Helper to register a profile point
[[maybe_unused]]
static inline bool register_profile_point(profile_point *pp)
{
    std::lock_guard<std::mutex> lock(registry_mutex);
    profile_point_registry.push_back(pp);
    return true;
}

// Keep this below a cache line size.
struct alignas(64) profile_point_percpu {
    unsigned long nr;
    unsigned long time_cycles;
    int num_samples;
    int sample_after;
    unsigned long time_samples[PP_SAMPLE_BUF_SIZE];
};

struct alignas(64) profile_point {
    // Keep per-cpu critical fields at the beginning of the struct, so they're always in the
    // same cache line.
    struct profile_point_percpu *percpu;
    bool enabled;
    bool measures_cpu_cycles;
    std::string pp_name;

    profile_point() : profile_point(std::string("")) {}

    profile_point(std::string name)
    {
        enabled = false;
        pp_name = name;

        // If represents_cycles is explicitly false or name ends with '_', use raw cycles
        // Otherwise, use the explicit parameter value
        bool ends_in_underscore = !name.empty() && name.back() == '_';
        this->measures_cpu_cycles = !ends_in_underscore;

        // Allocate percpu data
        percpu = new profile_point_percpu();
        memset(percpu, 0, sizeof(profile_point_percpu));
        percpu->sample_after =
            PP_SAMPLE_MIN_DELAY + (rand() % (PP_SAMPLE_MAX_DELAY - PP_SAMPLE_MIN_DELAY));
    }

    ~profile_point()
    {
        if (percpu) {
            delete percpu;
            percpu = nullptr;
        }
    }

    void clear()
    {
        enabled = false;
        percpu->nr = 0;
        percpu->time_cycles = 0;
        percpu->num_samples = 0;
    }

    void enable(bool b) { enabled = b; }

    // TODO: move the remaining helper functions into the PP class itself.
    inline void add(unsigned long x)
    {
        if (!enabled)
            return;

        percpu->nr++;
        percpu->time_cycles += x;
    }

    // TODO: move the remaining helper functions into the PP class itself.
    inline void increment() { this->add(1); }

    inline void record(unsigned long entry)
    {
        if (!enabled)
            return;

        percpu->nr++;
        percpu->time_cycles += entry;

        if (percpu->num_samples == PP_SAMPLE_BUF_SIZE)
            return;

        if (percpu->sample_after > 0) {
            percpu->sample_after--;
            return;
        }
        percpu->time_samples[percpu->num_samples] = entry;
        percpu->num_samples++;
        percpu->sample_after =
            PP_SAMPLE_MIN_DELAY + (rand() % (PP_SAMPLE_MAX_DELAY - PP_SAMPLE_MIN_DELAY));
    }

    inline void exit(const TimeStamp &start)
    {
        if (!enabled)
            return;

        TimeStamp __PP_end_time = GetTimestamp();
        TimeStamp __PP_diff_time = __PP_end_time - start;

        percpu->nr++;
        percpu->time_cycles += __PP_diff_time;

        if (percpu->num_samples == PP_SAMPLE_BUF_SIZE)
            return;

        if (percpu->sample_after > 0) {
            percpu->sample_after--;
            return;
        }
        percpu->time_samples[percpu->num_samples] = __PP_diff_time;
        percpu->num_samples++;
        percpu->sample_after =
            PP_SAMPLE_MIN_DELAY + (rand() % (PP_SAMPLE_MAX_DELAY - PP_SAMPLE_MIN_DELAY));
    }

    std::string toString() const
    {
        std::ostringstream oss;

        oss << "PP: " << pp_name << ", time_us: ";

        if (measures_cpu_cycles) {
            oss << TimeStamp_to_us(percpu->time_cycles);
        } else {
            oss << percpu->time_cycles;
        };
        oss << ", nr: " << percpu->nr << ", num_samples: " << percpu->num_samples << std::endl;

        if (percpu->num_samples > 0) {
            print_sample_buf(oss, percpu->time_samples, percpu->num_samples);
        }
        return oss.str();
    }

    nlohmann::json toJson() const
    {
        nlohmann::json j;
        j["pp_name"] = pp_name;

        if (measures_cpu_cycles) {
            j["total_time_us"] = TimeStamp_to_us(percpu->time_cycles);
        } else {
            j["total_time_us"] = percpu->time_cycles;
        }

        j["nr"] = percpu->nr;

        nlohmann::json samples = nlohmann::json::array();
        for (int i = 0; i < percpu->num_samples; i++) {
            if (measures_cpu_cycles) {
                samples.push_back(TimeStamp_to_us(percpu->time_samples[i]));
            } else {
                samples.push_back(percpu->time_samples[i]);
            }
        }
        j["samples"] = samples;
        return j;
    }

private:
    void print_sample_buf(std::ostringstream &oss, unsigned long *buf, int buf_size) const
    {
        // Print 12 numbers per line
        for (int i = 0; i < buf_size; i += 12) {
            oss << "\t";
            for (int j = 0; j < 12 && (i + j) < buf_size; j++) {
                if (measures_cpu_cycles) {
                    oss << TimeStamp_to_us(buf[i + j]) << " ";
                } else {
                    oss << buf[i + j] << " ";
                }
            }
            oss << std::endl;
        }
    }
};

// Variable Name mangling.
#define _PP_NAME(name) __profilepoint_##name
#define _PP_TIME(name) __profilepoint_start_ns_##name
#define _PP_BIN(name, low, high) name##_##low##_to_##high

#define _DEFINE_PP_VAR(name)                                                                       \
    struct profile_point _PP_NAME(name)(std::string(#name));                                       \
    static bool _PP_NAME(name##_registered) = register_profile_point(&_PP_NAME(name))

#define _DECLARE_PP_VAR(name) extern struct profile_point _PP_NAME(name)

/* ==========================================================
 * Profiling Point API
 * ==========================================================
 *
 * Typical usage:
 *     // in global environment
 *     DECLARE_PP(foo);
 *
 *     void f() {
 *         PP_STATE(foo);  // declares a tmp variable for profiling
 *
 *         PP_ENTER(foo); // begin profiling
 *         foo();
 *         PP_EXIT(foo);  // exit profiled region.
 *      }
 *
 * The region's runtime will be saved to profile point `foo`. Many other useful
 * statistics are recorded (such as samples for CDFs).
 *
 * We also present a lower-level API that gives you more control over what is timestamped.
 * (they are compiled out if profiling is disabled).
 *
 *     PP_NEW_TIMESTAMP(foo_time); // declare a struct timestamp with name `foo_time`.
 *     PP_SAVE_TIMESTAMP(foo_time);       // save the current timestamp to `foo_time`.
 *
 * Many other features are present (see API comments below).
 * Reach out to to Yash (yash@yashlala.com) if you're confused.
 */

/* LOW-LEVEL API: USE ONLY IF NEEDED */

/* Low-Level API: Declare a struct TimeStamp. */
#define PP_NEW_TIMESTAMP(name) TimeStamp name
/* Low-Level API: Save the current time to a struct TimeStamp. */
#define PP_SAVE_TIMESTAMP(name) name = GetTimestamp()
/* Low-Level API: shorthand for PP_NEW_TIMESTAMP + PP_SAVE_TIMESTAMP */
#define PP_SAVE_NEW_TIMESTAMP(name) TimeStamp name = GetTimestamp()

/* HIGH-LEVEL API: FOR GENERAL USE! */

/*
 * Define a profile point. Should be done once; in a C file.
 * Once a profile point is defined, it'll automatically be printed
 * in `print_pps()`.
 */
#define DEFINE_PP(name) _DEFINE_PP_VAR(name)

/*
 * Declare a profile point ("extern profile_point yee"). Should be invoked once in the files where
 * a PP is being used.
 */
#define DECLARE_PP(name) _DECLARE_PP_VAR(name)

/*
 * Declare one of these at the head of every function where you use a PP
 */
#define PP_STATE(name) [[maybe_unused]] TimeStamp _PP_TIME(name)

#define PP_ENTER(name)                                                                             \
    do {                                                                                           \
        _PP_TIME(name) = GetTimestamp();                                                           \
    } while (0)

// Exit from a globally defined profile point.
#define PP_EXIT(name) (_PP_NAME(name)).exit(_PP_TIME(name))

/*
 * Same as PP exit -- except instead of automatically sampling the current time, just record
 * a sample that the user provided.
 */
#define PP_RECORD(name, sample) (_PP_NAME(name)).record(sample)
#define PP_ADD(name, count) (_PP_NAME(name)).add(count)
#define PP_INCREMENT(name) (_PP_NAME(name)).increment()

#define _PP_SAMPLE(name, percpu, sample)                                                           \
    do {                                                                                           \
        if (!_PP_NAME(name).enabled)                                                               \
            break;                                                                                 \
                                                                                                   \
        percpu->nr++;                                                                              \
                                                                                                   \
        if (percpu->num_samples == PP_SAMPLE_BUF_SIZE)                                             \
            break;                                                                                 \
                                                                                                   \
        percpu->time_samples[percpu->num_samples] = sample;                                        \
        percpu->num_samples++;                                                                     \
    } while (0)

/*
 * Unconditionally log a sample into our PP's sample buffer. No timeout.
 */
#define PP_SAMPLE(name, sample) _PP_SAMPLE(name, _PP_NAME(name).percpu, sample)

// Run something only if PPs are enabled :)
#define PP_DO(code)                                                                                \
    do {                                                                                           \
        code;                                                                                      \
    } while (0);

// Deprecated...used to be used for reading per-cpu vars from different CPUs.
// But we removed the per-cpu var functionality anyways.
#define PP_READ_PER_CPU(name, field) (_PP_NAME(name).percpu->field)

#else // !CONFIG_PROFILING_POINTS
#define PP_PROFILING_ENABLED false

struct profile_point_percpu {
};
struct profile_point {
    profile_point() {}
    profile_point(std::string name, bool represents_cycles = false)
    {
        (void)name;
        (void)represents_cycles;
    }
    ~profile_point() {}

    void clear() {}
    void enable(bool b) { (void)b; }
    void add(unsigned long x) { (void)x; }
    void increment() {}
    void record(unsigned long entry) { (void)entry; }
    void exit(const TimeStamp &start) { (void)start; }
    std::string toString() const { return ""; }
    nlohmann::json toJson() const { return nlohmann::json::object(); }
};

// Variable Name mangling helpers
#define _PP_NAME(name) __profilepoint_##name
#define _PP_TIME(name) __profilepoint_start_ns_##name

/* ==========================================================
 * Profiling Point API
 * ==========================================================
 */

#define PP_SAVE_TIMESTAMP(name)                                                                    \
    do {                                                                                           \
    } while (0)
/* Low-Level API: Declare a struct TimeStamp. */
#define PP_NEW_TIMESTAMP(name) constexpr TimeStamp name = 0
/* Low-Level API: Save the current time to a struct TimeStamp. */
#define PP_SAVE_NEW_TIMESTAMP(name) PP_NEW_TIMESTAMP(name)

#define DEFINE_PP(name)
#define DECLARE_PP(name)
#define PP_STATE(name)                                                                             \
    do {                                                                                           \
    } while (0)
#define PP_ENTER(name)                                                                             \
    do {                                                                                           \
    } while (0)
#define PP_EXIT(name)                                                                              \
    do {                                                                                           \
    } while (0)
#define PP_RECORD(name, sample)                                                                    \
    do {                                                                                           \
    } while (0)
#define PP_ADD(name, count)                                                                        \
    do {                                                                                           \
    } while (0)
#define PP_INCREMENT(name)                                                                         \
    do {                                                                                           \
    } while (0)
#define PP_SAMPLE(name, sample)                                                                    \
    do {                                                                                           \
    } while (0)
#define PP_READ_PER_CPU(name, field) (-1L)
#define PP_DO(code)                                                                                \
    do {                                                                                           \
    } while (0)

inline void print_pps() {}
inline void clear_pps() {}
inline void enable_pps(bool mode) { (void)mode; }
inline nlohmann::json dump_pp_json() { return nlohmann::json::array(); }

#endif // !CONFIG_PROFILING_POINTS

#endif // PROFILING_POINTS_H
