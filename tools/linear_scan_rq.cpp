// linear_scan_rq.cpp
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// Number of dimensions per point
constexpr size_t NUM_DIMENSIONS = 16;

// Read a CSV of 64-bit integers into a vector of points
std::vector<std::vector<uint64_t>> load_points(const std::string &path)
{
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("Cannot open data file: " + path);
    std::vector<std::vector<uint64_t>> pts;
    std::string line;
    while (std::getline(in, line)) {
        std::vector<uint64_t> row;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            row.push_back(std::stoull(cell));
        }
        if (row.size() != NUM_DIMENSIONS)
            throw std::runtime_error("Expected " + std::to_string(NUM_DIMENSIONS) + " dims, got " +
                                     std::to_string(row.size()));
        pts.push_back(std::move(row));
    }
    return pts;
}

// Read matching lower/upper CSV into a vector of ranges
using Range = std::pair<std::vector<uint64_t>, std::vector<uint64_t>>;
std::vector<Range> load_ranges(const std::string &low_path, const std::string &high_path)
{
    std::ifstream lowf(low_path), highf(high_path);
    if (!lowf || !highf)
        throw std::runtime_error("Cannot open range files");
    std::vector<Range> ranges;
    std::string lline, hline;
    while (std::getline(lowf, lline) && std::getline(highf, hline)) {
        std::vector<uint64_t> lo, hi;
        {
            std::stringstream ss(lline);
            std::string cell;
            while (std::getline(ss, cell, ','))
                lo.push_back(std::stoull(cell));
        }
        {
            std::stringstream ss(hline);
            std::string cell;
            while (std::getline(ss, cell, ','))
                hi.push_back(std::stoull(cell));
        }
        if (lo.size() != NUM_DIMENSIONS || hi.size() != NUM_DIMENSIONS)
            throw std::runtime_error("Range dimension mismatch");
        ranges.emplace_back(std::move(lo), std::move(hi));
    }
    return ranges;
}

// Perform linear scan for a single query
void scan_query(const std::vector<std::vector<uint64_t>> &data, const Range &range,
                std::vector<std::vector<uint64_t>> &out)
{
    const auto &lo = range.first;
    const auto &hi = range.second;
    for (auto &pt : data) {
        bool inside = true;
        for (size_t d = 0; d < NUM_DIMENSIONS; ++d) {
            if (pt[d] < lo[d] || pt[d] > hi[d]) {
                inside = false;
                break;
            }
        }
        if (inside)
            out.push_back(pt);
    }
}

int main(int argc, char **argv)
{
    if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        std::cerr << "Usage: " << argv[0]
                  << " <data.csv> <lower.csv> <upper.csv> <out.txt> [threads]\n"
                  << " if [threads] <=1, runs single-threaded; else splits "
                     "queries evenly.\n";
        return 0;
    }

    if (argc < 5 || argc > 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <data.csv> <lower.csv> <upper.csv> <out.txt> [threads]\n"
                  << " if [threads] <=1, runs single-threaded; else splits "
                     "queries evenly.\n";
        return 1;
    }
    std::string data_file = argv[1];
    std::string lower_file = argv[2];
    std::string upper_file = argv[3];
    std::string output_file = argv[4];
    int threads = 1;
    if (argc == 6)
        threads = std::max(1, std::stoi(argv[5]));

    // 1) Load everything
    auto data = load_points(data_file);
    auto ranges = load_ranges(lower_file, upper_file);
    size_t Q = ranges.size();

    // 2) Prepare results container
    std::vector<std::vector<std::vector<uint64_t>>> results(Q);

    // 3) Worker lambda for a contiguous slice of queries
    auto worker = [&](size_t qstart, size_t qend) {
        for (size_t q = qstart; q < qend; ++q) {
            scan_query(data, ranges[q], results[q]);
        }
    };

    // 4) Launch threads
    std::vector<std::thread> pool;
    if (threads <= 1) {
        worker(0, Q);
    } else {
        for (int t = 0; t < threads; ++t) {
            size_t start = (uint64_t)t * Q / threads;
            size_t end = (uint64_t)(t + 1) * Q / threads;
            pool.emplace_back(worker, start, end);
        }
        for (auto &th : pool)
            th.join();
    }

    // 5) Write output in-order
    std::ofstream out(output_file);
    if (!out) {
        std::cerr << "Cannot create output file\n";
        return 1;
    }
    for (size_t q = 0; q < Q; ++q) {
        auto &pts = results[q];
        out << "[id: " << q << ", count: " << pts.size() << "]\n";
        for (auto &pt : pts) {
            for (size_t d = 0; d < NUM_DIMENSIONS; ++d) {
                out << pt[d] << (d + 1 < NUM_DIMENSIONS ? "," : "\n");
            }
        }
        out << "\n";
    }

    std::cout << "Done. Queried " << Q << " ranges over " << data.size() << " points using "
              << threads << (threads > 1 ? " threads.\n" : " thread.\n");
    return 0;
}
