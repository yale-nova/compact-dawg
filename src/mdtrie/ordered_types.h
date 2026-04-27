#ifndef ORDERED_TYPES_H_
#define ORDERED_TYPES_H_

#include <cstdint>
#include <cstring> // std::memcpy

// GLOBAL PROGRAM TYPES

// Represents a _single coordinate value_ in one dimension.
// ie: `struct data point { coordinate_t coordinates[D]; }` where D is a `dimension_count_t`. So
// `n_dimensions_t` is _number_ of dimensions...but `coordinate_t` is the dimension itself.
//
// (Our trie doesn't store arbitrary data yet...it just stores vectors of `coordinate_t`s).
typedef uint32_t coordinate_t;

// Used only for the _ordered u32_ representation of a float, so the compiler will
// warn us if we mix them up. TODO(yash): gradually convert existing code to use
// this for safety.
typedef uint32_t ordered_coordinate_t;

// TYPE CONVERSION HELPERS

namespace ordered_types
{

// Helper: convert float to ordered uint32 key (makes floats sortable as
// unsigned integers)
ordered_coordinate_t float_to_ordered_u32(float f)
{
    ordered_coordinate_t u;
    static_assert(sizeof(u) == sizeof(f), "float/ordered_coordinate_t size mismatch");
    std::memcpy(&u, &f, sizeof(u)); // Get raw IEEE 754 bits
    if (u & 0x80000000u)            // If negative (sign bit is 1)
        return ~u;                  // Flip all bits
    else                            // If positive or zero
        return u ^ 0x80000000u;     // Flip only the sign bit
}

// Helper: convert ordered uint32 (which makes floats sortable as integers)
// back to a float representation.
float ordered_u32_to_float(ordered_coordinate_t u)
{
    float f;
    static_assert(sizeof(u) == sizeof(f), "float/ordered_coordinate_t size mismatch");
    if (u & 0x80000000u)  // MSB = 1 means original float was positive
        u ^= 0x80000000u; // Flip sign bit back
    else                  // MSB = 0 means was negative
        u = ~u;           // Flip all bits back

    std::memcpy(&f, &u, sizeof(f));
    return f;
}
}

#endif // ORDERED_TYPES_H_
