#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <dawgdic/dictionary.h>

int main(int argc, char **argv)
{
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <dawg_dic_file> <start_key> <end_key>" << std::endl;
        return 1;
    }

    const char *dawg_path = argv[1];
    const char *start_key = argv[2];
    const char *end_key = argv[3];

    // Load the dictionary
    dawgdic::Dictionary dic;
    std::ifstream dawg_file(dawg_path, std::ios::binary);
    if (!dawg_file) {
        std::cerr << "Error: Could not open dictionary file: " << dawg_path << std::endl;
        return 1;
    }

    if (!dic.Read(&dawg_file)) {
        std::cerr << "Error: Failed to read dictionary from " << dawg_path << std::endl;
        dawg_file.close();
        return 1;
    }
    dawg_file.close();

    std::cout << "Dictionary loaded successfully. Size: " << dic.size() << " units." << std::endl;

    std::vector<std::string> results;

    dic.RangeSearch(start_key, end_key, &results);

    std::cout << "\nFound " << results.size() << " keys:" << std::endl;
    for (const auto &key : results) {
        std::cout << key << std::endl;
    }

    return 0;
}
