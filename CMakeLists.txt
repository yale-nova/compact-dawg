cmake_minimum_required(VERSION 3.14)
project(CompactDawg LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_definitions(CONFIG_PROFILING_POINTS)

add_library(profiling_points STATIC src/mdtrie/profiling_points.cpp)
target_include_directories(profiling_points PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/lib)

add_library(compact_dawg_headers INTERFACE)
target_include_directories(compact_dawg_headers INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/lib
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/tools)
target_compile_options(compact_dawg_headers INTERFACE
    -mbmi2
    -mpopcnt
    -mavx2
    -O3
    -g3
    -fno-omit-frame-pointer
    -Wno-class-memaccess
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wunreachable-code)

include(CTest)
enable_testing()

add_executable(test_compact_dawg_core tests/test_compact_dawg_core.cpp)
target_link_libraries(test_compact_dawg_core PRIVATE compact_dawg_headers profiling_points)
add_test(NAME CompactDawgCoreTest COMMAND test_compact_dawg_core)

add_executable(test_dynamic_dawg tests/test_dynamic_dawg.cpp)
target_link_libraries(test_dynamic_dawg PRIVATE compact_dawg_headers profiling_points)
add_test(NAME DynamicDawgTest COMMAND test_dynamic_dawg)

add_executable(test_compact_dawg_spatial_1024d tests/test_compact_dawg_spatial_1024d.cpp)
target_link_libraries(test_compact_dawg_spatial_1024d PRIVATE compact_dawg_headers profiling_points)
add_test(NAME CompactDawgSpatial1024dTest COMMAND test_compact_dawg_spatial_1024d)

add_executable(test_dawgdic_lexicographic tests/test_dawgdic_lexicographic.cpp)
target_link_libraries(test_dawgdic_lexicographic PRIVATE compact_dawg_headers profiling_points)
add_test(
    NAME DawgdicLexicographicTest
    COMMAND test_dawgdic_lexicographic
            ${CMAKE_SOURCE_DIR}/tests/testdata/test.dawg
            apple
            banana)

add_executable(bench_dawg_storage tools/bench_dawg_storage.cpp)
target_link_libraries(bench_dawg_storage PRIVATE compact_dawg_headers profiling_points)

add_executable(bench_dawg_sweep tools/bench_dawg_sweep.cpp)
target_link_libraries(bench_dawg_sweep PRIVATE compact_dawg_headers profiling_points)

add_executable(bench_dawg_rq tools/bench_dawg_rq.cpp)
target_link_libraries(bench_dawg_rq PRIVATE compact_dawg_headers profiling_points)

add_executable(bench_dawg_sharing tools/bench_dawg_sharing.cpp)
target_link_libraries(bench_dawg_sharing PRIVATE compact_dawg_headers profiling_points)

add_executable(bench_dynamic_dawg_sweep tools/bench_dynamic_dawg_sweep.cpp)
target_link_libraries(bench_dynamic_dawg_sweep PRIVATE compact_dawg_headers profiling_points)

add_executable(analyze_bitstring_structure tools/analyze_bitstring_structure.cpp)
target_link_libraries(analyze_bitstring_structure PRIVATE compact_dawg_headers profiling_points)

add_executable(gen_rq_queries_exact10_1024d tools/gen_rq_queries_exact10_1024d.cpp)
target_link_libraries(gen_rq_queries_exact10_1024d PRIVATE compact_dawg_headers profiling_points)

add_executable(linear_scan_rq tools/linear_scan_rq.cpp)
target_link_libraries(linear_scan_rq PRIVATE compact_dawg_headers profiling_points)
