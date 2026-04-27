#ifndef DAWGDIC_PACKED_DICTIONARY_UNIT_H
#define DAWGDIC_PACKED_DICTIONARY_UNIT_H

#include "base-types.h"

namespace dawgdic {

template <int LABEL_BITS, int OFFSET_BITS>
class PackedDictionaryUnit {
 public:
  static constexpr int TOTAL_BITS = 1 + 1 + LABEL_BITS + OFFSET_BITS;
  
  static constexpr uint64_t LABEL_MASK = (1ULL << LABEL_BITS) - 1;
  static constexpr uint64_t OFFSET_MASK = (1ULL << OFFSET_BITS) - 1;
  static constexpr uint64_t IS_LEAF_BIT = 1ULL << (TOTAL_BITS - 1);
  static constexpr uint64_t HAS_LEAF_BIT = 1ULL << (LABEL_BITS);

  using LabelType = typename std::conditional<(LABEL_BITS > 16), uint32_t,
                      typename std::conditional<(LABEL_BITS > 8), uint16_t, unsigned char>::type>::type;

  explicit PackedDictionaryUnit(uint64_t data) : data_(data) {}

  bool is_leaf() const {
    return (data_ & IS_LEAF_BIT) != 0;
  }

  bool has_leaf() const {
    return (data_ & HAS_LEAF_BIT) != 0;
  }

  LabelType label() const {
    return static_cast<LabelType>(data_ & LABEL_MASK);
  }

  BaseType offset() const {
    return static_cast<BaseType>((data_ >> (LABEL_BITS + 1)) & OFFSET_MASK);
  }

  ValueType value() const {
    return static_cast<ValueType>(data_ & ~IS_LEAF_BIT);
  }

  static uint64_t make_leaf(ValueType value) {
    return IS_LEAF_BIT | (static_cast<uint64_t>(value) & ~IS_LEAF_BIT);
  }

  static uint64_t make_node(BaseType offset, LabelType label, bool has_leaf) {
    uint64_t val = static_cast<uint64_t>(label) & LABEL_MASK;
    if (has_leaf) {
      val |= HAS_LEAF_BIT;
    }
    val |= (static_cast<uint64_t>(offset) & OFFSET_MASK) << (LABEL_BITS + 1);
    return val;
  }

 private:
  uint64_t data_;
};

}

#endif // DAWGDIC_PACKED_DICTIONARY_UNIT_H
