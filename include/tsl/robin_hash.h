/**
 * MIT License
 *
 * Copyright (c) 2017 Thibaut Goetghebuer-Planchon <tessil@gmx.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifndef TSL_ROBIN_HASH_H
#define TSL_ROBIN_HASH_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iterator>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "robin_growth_policy.h"

namespace tsl {

namespace detail_robin_hash {

template <typename T>
struct make_void {
  using type = void;
};

template <typename T, typename = void>
struct has_is_transparent : std::false_type {};

template <typename T>
struct has_is_transparent<T,
                          typename make_void<typename T::is_transparent>::type>
    : std::true_type {};

template <typename U>
struct is_power_of_two_policy : std::false_type {};

template <std::size_t GrowthFactor>
struct is_power_of_two_policy<tsl::rh::power_of_two_growth_policy<GrowthFactor>>
    : std::true_type {};

template <typename T, typename U>
static T numeric_cast(U value,
                      const char* error_message = "numeric_cast() failed.") {
  T ret = static_cast<T>(value);
  if (static_cast<U>(ret) != value) {
    TSL_RH_THROW_OR_TERMINATE(std::runtime_error, error_message);
  }

  const bool is_same_signedness =
      (std::is_unsigned<T>::value && std::is_unsigned<U>::value) ||
      (std::is_signed<T>::value && std::is_signed<U>::value);
  if (!is_same_signedness && (ret < T{}) != (value < U{})) {
    TSL_RH_THROW_OR_TERMINATE(std::runtime_error, error_message);
  }

  TSL_RH_UNUSED(error_message);

  return ret;
}

template <class T, class Deserializer>
static T deserialize_value(Deserializer& deserializer) {
  // MSVC < 2017 is not conformant, circumvent the problem by removing the
  // template keyword
#if defined(_MSC_VER) && _MSC_VER < 1910
  return deserializer.Deserializer::operator()<T>();
#else
  return deserializer.Deserializer::template operator()<T>();
#endif
}

/**
 * Fixed size type used to represent size_type values on serialization. Need to
 * be big enough to represent a std::size_t on 32 and 64 bits platforms, and
 * must be the same size on both platforms.
 */
using slz_size_type = std::uint64_t;
static_assert(std::numeric_limits<slz_size_type>::max() >=
                  std::numeric_limits<std::size_t>::max(),
              "slz_size_type must be >= std::size_t");

using truncated_hash_type = std::uint32_t;

/**
 * Helper class that stores a truncated hash if StoreHash is true and nothing
 * otherwise.
 */
template <bool StoreHash>
class bucket_entry_hash {
 public:
  bool bucket_hash_equal(std::size_t /*hash*/) const noexcept { return true; }

  truncated_hash_type truncated_hash() const noexcept { return 0; }

 protected:
  void set_hash(truncated_hash_type /*hash*/) noexcept {}
};

template <>
class bucket_entry_hash<true> {
 public:
  bool bucket_hash_equal(std::size_t hash) const noexcept {
    return m_hash == truncated_hash_type(hash);
  }

  truncated_hash_type truncated_hash() const noexcept { return m_hash; }

 protected:
  void set_hash(truncated_hash_type hash) noexcept {
    m_hash = truncated_hash_type(hash);
  }

 private:
  truncated_hash_type m_hash;
};

/**
 * Each bucket entry has:
 * - A value of type `ValueType`.
 * - An integer to store how far the value of the bucket, if any, is from its
 * ideal bucket (ex: if the current bucket 5 has the value 'foo' and
 * `hash('foo') % nb_buckets` == 3, `dist_from_ideal_bucket()` will return 2 as
 * the current value of the bucket is two buckets away from its ideal bucket) If
 * there is no value in the bucket (i.e. `empty()` is true)
 * `dist_from_ideal_bucket()` will be < 0.
 * - A marker which tells us if the bucket is the last bucket of the bucket
 * array (useful for the iterator of the hash table).
 * - If `StoreHash` is true, 32 bits of the hash of the value, if any, are also
 * stored in the bucket. If the size of the hash is more than 32 bits, it is
 * truncated. We don't store the full hash as storing the hash is a potential
 * opportunity to use the unused space due to the alignment of the bucket_entry
 * structure. We can thus potentially store the hash without any extra space
 *   (which would not be possible with 64 bits of the hash).
 */
template <typename ValueType, bool StoreHash>
class bucket_entry : public bucket_entry_hash<StoreHash> {
  using bucket_hash = bucket_entry_hash<StoreHash>;

 public:
  using value_type = ValueType;
  using distance_type = std::int16_t;

  bucket_entry() noexcept
      : bucket_hash(),
        m_dist_from_ideal_bucket(EMPTY_MARKER_DIST_FROM_IDEAL_BUCKET),
        m_last_bucket(false) {
    tsl_rh_assert(empty());
  }

  bucket_entry(bool last_bucket) noexcept
      : bucket_hash(),
        m_dist_from_ideal_bucket(EMPTY_MARKER_DIST_FROM_IDEAL_BUCKET),
        m_last_bucket(last_bucket) {
    tsl_rh_assert(empty());
  }

  bucket_entry(const bucket_entry& other) noexcept(
      std::is_nothrow_copy_constructible<value_type>::value)
      : bucket_hash(other),
        m_dist_from_ideal_bucket(EMPTY_MARKER_DIST_FROM_IDEAL_BUCKET),
        m_last_bucket(other.m_last_bucket) {
    if (!other.empty()) {
      ::new (static_cast<void*>(std::addressof(m_value)))
          value_type(other.value());
      m_dist_from_ideal_bucket = other.m_dist_from_ideal_bucket;
    }
    tsl_rh_assert(empty() == other.empty());
  }

  /**
   * Never really used, but still necessary as we must call resize on an empty
   * `std::vector<bucket_entry>`. and we need to support move-only types. See
   * robin_hash constructor for details.
   */
  bucket_entry(bucket_entry&& other) noexcept(
      std::is_nothrow_move_constructible<value_type>::value)
      : bucket_hash(std::move(other)),
        m_dist_from_ideal_bucket(EMPTY_MARKER_DIST_FROM_IDEAL_BUCKET),
        m_last_bucket(other.m_last_bucket) {
    if (!other.empty()) {
      ::new (static_cast<void*>(std::addressof(m_value)))
          value_type(std::move(other.value()));
      m_dist_from_ideal_bucket = other.m_dist_from_ideal_bucket;
    }
    tsl_rh_assert(empty() == other.empty());
  }

  bucket_entry& operator=(const bucket_entry& other) noexcept(
      std::is_nothrow_copy_constructible<value_type>::value) {
    if (this != &other) {
      clear();

      bucket_hash::operator=(other);
      if (!other.empty()) {
        ::new (static_cast<void*>(std::addressof(m_value)))
            value_type(other.value());
      }

      m_dist_from_ideal_bucket = other.m_dist_from_ideal_bucket;
      m_last_bucket = other.m_last_bucket;
    }

    return *this;
  }

  bucket_entry& operator=(bucket_entry&&) = delete;

  ~bucket_entry() noexcept { clear(); }

  void clear() noexcept {
    if (!empty()) {
      destroy_value();
      m_dist_from_ideal_bucket = EMPTY_MARKER_DIST_FROM_IDEAL_BUCKET;
    }
  }

  bool empty() const noexcept {
    return m_dist_from_ideal_bucket == EMPTY_MARKER_DIST_FROM_IDEAL_BUCKET;
  }

  value_type& value() {
    tsl_rh_safe_assert(!empty(), "Attempt to access value of empty bucket");
    return *std::launder(
        reinterpret_cast<value_type*>(std::addressof(m_value)));
  }

  const value_type& value() const {
    tsl_rh_safe_assert(!empty(), "Attempt to access value of empty bucket");
    return *std::launder(
        reinterpret_cast<const value_type*>(std::addressof(m_value)));
  }

  distance_type dist_from_ideal_bucket() const noexcept {
    return m_dist_from_ideal_bucket;
  }

  bool last_bucket() const noexcept { return m_last_bucket; }

  void set_as_last_bucket() noexcept { m_last_bucket = true; }

  template <typename... Args>
  void set_value_of_empty_bucket(distance_type dist_from_ideal_bucket,
                                 truncated_hash_type hash,
                                 Args&&... value_type_args) {
    tsl_rh_assert(dist_from_ideal_bucket >= 0);
    tsl_rh_assert(empty());

    ::new (static_cast<void*>(std::addressof(m_value)))
        value_type(std::forward<Args>(value_type_args)...);
    this->set_hash(hash);
    m_dist_from_ideal_bucket = dist_from_ideal_bucket;

    tsl_rh_assert(!empty());
  }

  void swap_with_value_in_bucket(distance_type& dist_from_ideal_bucket,
                                 truncated_hash_type& hash, value_type& value) {
    tsl_rh_assert(!empty());
    tsl_rh_assert(dist_from_ideal_bucket > m_dist_from_ideal_bucket);

    using std::swap;
    swap(value, this->value());
    swap(dist_from_ideal_bucket, m_dist_from_ideal_bucket);

    if (StoreHash) {
      const truncated_hash_type tmp_hash = this->truncated_hash();
      this->set_hash(hash);
      hash = tmp_hash;
    } else {
      // Avoid warning of unused variable if StoreHash is false
      TSL_RH_UNUSED(hash);
    }
  }

  static truncated_hash_type truncate_hash(std::size_t hash) noexcept {
    return truncated_hash_type(hash);
  }

 private:
  void destroy_value() noexcept {
    tsl_rh_assert(!empty());
    value().~value_type();
  }

 public:
  static const distance_type EMPTY_MARKER_DIST_FROM_IDEAL_BUCKET = -1;
  static const distance_type DIST_FROM_IDEAL_BUCKET_LIMIT = 8192;
  static_assert(DIST_FROM_IDEAL_BUCKET_LIMIT <=
                    std::numeric_limits<distance_type>::max() - 1,
                "DIST_FROM_IDEAL_BUCKET_LIMIT must be <= "
                "std::numeric_limits<distance_type>::max() - 1.");

 private:
  distance_type m_dist_from_ideal_bucket;
  bool m_last_bucket;
  alignas(value_type) unsigned char m_value[sizeof(value_type)];
};

/**
 * Internal common class used by `robin_map` and `robin_set`.
 *
 * ValueType is what will be stored by `robin_hash` (usually `std::pair<Key, T>`
 * for map and `Key` for set).
 *
 * `KeySelect` should be a `FunctionObject` which takes a `ValueType` in
 * parameter and returns a reference to the key.
 *
 * `ValueSelect` should be a `FunctionObject` which takes a `ValueType` in
 * parameter and returns a reference to the value. `ValueSelect` should be void
 * if there is no value (in a set for example).
 *
 * The strong exception guarantee only holds if the expression
 * `std::is_nothrow_swappable<ValueType>::value &&
 * std::is_nothrow_move_constructible<ValueType>::value` is true.
 *
 * Behaviour is undefined if the destructor of `ValueType` throws.
 */
template <class ValueType, class KeySelect, class ValueSelect, class Hash,
          class KeyEqual, class Allocator, bool StoreHash, class GrowthPolicy>
class robin_hash : private Hash, private KeyEqual, private GrowthPolicy {
 private:
  template <typename U>
  using has_mapped_type =
      typename std::integral_constant<bool, !std::is_same<U, void>::value>;

  static_assert(
      noexcept(std::declval<GrowthPolicy>().bucket_for_hash(std::size_t(0))),
      "GrowthPolicy::bucket_for_hash must be noexcept.");
  static_assert(noexcept(std::declval<GrowthPolicy>().clear()),
                "GrowthPolicy::clear must be noexcept.");

 public:
  template <bool IsConst>
  class robin_iterator;

  using key_type = typename KeySelect::key_type;
  using value_type = ValueType;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using hasher = Hash;
  using key_equal = KeyEqual;
  using allocator_type = Allocator;
  using reference = value_type&;
  using const_reference = const value_type&;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  using iterator = robin_iterator<false>;
  using const_iterator = robin_iterator<true>;

 private:
  /**
   * Either store the hash because we are asked by the `StoreHash` template
   * parameter or store the hash because it doesn't cost us anything in size and
   * can be used to speed up rehash.
   */
  static constexpr bool STORE_HASH =
      StoreHash ||
      ((sizeof(tsl::detail_robin_hash::bucket_entry<value_type, true>) ==
        sizeof(tsl::detail_robin_hash::bucket_entry<value_type, false>)) &&
       (sizeof(std::size_t) == sizeof(truncated_hash_type) ||
        is_power_of_two_policy<GrowthPolicy>::value) &&
       // Don't store the hash for primitive types with default hash.
       (!std::is_arithmetic<key_type>::value ||
        !std::is_same<Hash, std::hash<key_type>>::value));

  /**
   * Only use the stored hash on lookup if we are explicitly asked. We are not
   * sure how slow the KeyEqual operation is. An extra comparison may slow
   * things down with a fast KeyEqual.
   */
  static constexpr bool USE_STORED_HASH_ON_LOOKUP = StoreHash;

  /**
   * We can only use the hash on rehash if the size of the hash type is the same
   * as the stored one or if we use a power of two modulo. In the case of the
   * power of two modulo, we just mask the least significant bytes, we just have
   * to check that the truncated_hash_type didn't truncated more bytes.
   */
  static bool USE_STORED_HASH_ON_REHASH(size_type bucket_count) {
    if (STORE_HASH && sizeof(std::size_t) == sizeof(truncated_hash_type)) {
      TSL_RH_UNUSED(bucket_count);
      return true;
    } else if (STORE_HASH && is_power_of_two_policy<GrowthPolicy>::value) {
      return bucket_count == 0 ||
             (bucket_count - 1) <=
                 std::numeric_limits<truncated_hash_type>::max();
    } else {
      TSL_RH_UNUSED(bucket_count);
      return false;
    }
  }

  using bucket_entry =
      tsl::detail_robin_hash::bucket_entry<value_type, STORE_HASH>;
  using distance_type = typename bucket_entry::distance_type;

  using buckets_allocator = typename std::allocator_traits<
      allocator_type>::template rebind_alloc<bucket_entry>;
  using buckets_container_type = std::vector<bucket_entry, buckets_allocator>;

 public:
  /**
   * The 'operator*()' and 'operator->()' methods return a const reference and
   * const pointer respectively to the stored value type.
   *
   * In case of a map, to get a mutable reference to the value associated to a
   * key (the '.second' in the stored pair), you have to call 'value()'.
   *
   * The main reason for this is that if we returned a `std::pair<Key, T>&`
   * instead of a `const std::pair<Key, T>&`, the user may modify the key which
   * will put the map in a undefined state.
   */
  template <bool IsConst>
  class robin_iterator {
    friend class robin_hash;

   private:
    using bucket_entry_ptr =
        typename std::conditional<IsConst, const bucket_entry*,
                                  bucket_entry*>::type;

    robin_iterator(bucket_entry_ptr bucket) noexcept : m_bucket(bucket) {}

   public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = const typename robin_hash::value_type;
    using difference_type = std::ptrdiff_t;
    using reference = value_type&;
    using pointer = value_type*;

    robin_iterator() noexcept {}

    // Copy constructor from iterator to const_iterator.
    template <bool TIsConst = IsConst,
              typename std::enable_if<TIsConst>::type* = nullptr>
    robin_iterator(const robin_iterator<!TIsConst>& other) noexcept
        : m_bucket(other.m_bucket) {}

    robin_iterator(const robin_iterator& other) = default;
    robin_iterator(robin_iterator&& other) = default;
    robin_iterator& operator=(const robin_iterator& other) = default;
    robin_iterator& operator=(robin_iterator&& other) = default;

    const typename robin_hash::key_type& key() const {
        tsl_rh_safe_assert(m_bucket != nullptr, "Attempt to access key from null iterator");
        tsl_rh_safe_assert(!m_bucket->empty(), "Attempt to access key from iterator to empty bucket");
        return KeySelect()(m_bucket->value());
    }

    template <class U = ValueSelect,
              typename std::enable_if<has_mapped_type<U>::value &&
                                      IsConst>::type* = nullptr>
    const typename U::value_type& value() const {
        tsl_rh_safe_assert(m_bucket != nullptr, "Attempt to access value from null iterator");
        tsl_rh_safe_assert(!m_bucket->empty(), "Attempt to access value from iterator to empty bucket");
        return U()(m_bucket->value());
    }

    template <class U = ValueSelect,
              typename std::enable_if<has_mapped_type<U>::value &&
                                      !IsConst>::type* = nullptr>
    typename U::value_type& value() const {
        tsl_rh_safe_assert(m_bucket != nullptr, "Attempt to access value from null iterator");
        tsl_rh_safe_assert(!m_bucket->empty(), "Attempt to access value from iterator to empty bucket");
        return U()(m_bucket->value());
    }

    reference operator*() const {
        tsl_rh_safe_assert(m_bucket != nullptr, "Attempt to dereference null iterator");
        tsl_rh_safe_assert(!m_bucket->empty() || m_bucket->last_bucket(), "Attempt to dereference iterator to empty bucket");
        return m_bucket->value();
    }

    pointer operator->() const {
        tsl_rh_safe_assert(m_bucket != nullptr, "Attempt to dereference null iterator");
        tsl_rh_safe_assert(!m_bucket->empty() || m_bucket->last_bucket(), "Attempt to dereference iterator to empty bucket");
        return std::addressof(m_bucket->value());
    }

    robin_iterator& operator++() {
        tsl_rh_safe_assert(m_bucket != nullptr, "Attempt to increment null iterator");
        tsl_rh_safe_assert(!m_bucket->last_bucket(), "Attempt to increment end iterator");
        
        // 防止无限循环：跟踪已检查的桶数量
        const bucket_entry_ptr start_bucket = m_bucket;
        const std::size_t max_steps = 10000; // 合理的最大步数限制
        std::size_t steps = 0;
        
        while (true) {
            // 检查是否达到最大步数，防止无限循环
            tsl_rh_safe_assert(steps < max_steps, "Iterator increment reached maximum steps, possible infinite loop");
            steps++;
            
            if (m_bucket->last_bucket()) {
                ++m_bucket;
                return *this;
            }

            ++m_bucket;
            
            // 检查是否回到起始位置，防止循环
            tsl_rh_safe_assert(m_bucket != start_bucket, "Iterator increment loop detected");
            
            if (!m_bucket->empty()) {
                return *this;
            }
        }
    }

    robin_iterator operator++(int) {
      robin_iterator tmp(*this);
      ++*this;

      return tmp;
    }

    friend bool operator==(const robin_iterator& lhs,
                           const robin_iterator& rhs) {
      return lhs.m_bucket == rhs.m_bucket;
    }

    friend bool operator!=(const robin_iterator& lhs,
                           const robin_iterator& rhs) {
      return !(lhs == rhs);
    }

   private:
    bucket_entry_ptr m_bucket;
  };

 public:
  robin_hash(size_type bucket_count, const Hash& hash, const KeyEqual& equal,
             const Allocator& alloc,
             float min_load_factor = DEFAULT_MIN_LOAD_FACTOR,
             float max_load_factor = DEFAULT_MAX_LOAD_FACTOR)
      : Hash(hash),
        KeyEqual(equal),
        GrowthPolicy(bucket_count),
        m_buckets_data(bucket_count, alloc),
        m_buckets(m_buckets_data.empty() ? static_empty_bucket_ptr()
                                         : m_buckets_data.data()),
        m_bucket_count(bucket_count),
        m_nb_elements(0),
        m_grow_on_next_insert(false),
        m_try_shrink_on_next_insert(false) {
    if (bucket_count > max_bucket_count()) {
      TSL_RH_THROW_OR_TERMINATE(std::length_error,
                                "The map exceeds its maximum bucket count.");
    }

    if (m_bucket_count > 0) {
      tsl_rh_assert(!m_buckets_data.empty());
      m_buckets_data.back().set_as_last_bucket();
    }

    this->min_load_factor(min_load_factor);
    this->max_load_factor(max_load_factor);
  }

  robin_hash(const robin_hash& other)
      : Hash(other),
        KeyEqual(other),
        GrowthPolicy(other),
        m_buckets_data(other.m_buckets_data),
        m_buckets(m_buckets_data.empty() ? static_empty_bucket_ptr()
                                         : m_buckets_data.data()),
        m_bucket_count(other.m_bucket_count),
        m_nb_elements(other.m_nb_elements),
        m_load_threshold(other.m_load_threshold),
        m_min_load_factor(other.m_min_load_factor),
        m_max_load_factor(other.m_max_load_factor),
        m_grow_on_next_insert(other.m_grow_on_next_insert),
        m_try_shrink_on_next_insert(other.m_try_shrink_on_next_insert) {}

  robin_hash(robin_hash&& other) noexcept(
      std::is_nothrow_move_constructible<
          Hash>::value&& std::is_nothrow_move_constructible<KeyEqual>::value&&
          std::is_nothrow_move_constructible<GrowthPolicy>::value&&
              std::is_nothrow_move_constructible<buckets_container_type>::value)
      : Hash(std::move(static_cast<Hash&>(other))),
        KeyEqual(std::move(static_cast<KeyEqual&>(other))),
        GrowthPolicy(std::move(static_cast<GrowthPolicy&>(other))),
        m_buckets_data(std::move(other.m_buckets_data)),
        m_buckets(m_buckets_data.empty() ? static_empty_bucket_ptr()
                                         : m_buckets_data.data()),
        m_bucket_count(other.m_bucket_count),
        m_nb_elements(other.m_nb_elements),
        m_load_threshold(other.m_load_threshold),
        m_min_load_factor(other.m_min_load_factor),
        m_max_load_factor(other.m_max_load_factor),
        m_grow_on_next_insert(other.m_grow_on_next_insert),
        m_try_shrink_on_next_insert(other.m_try_shrink_on_next_insert) {
    other.clear_and_shrink();
  }

  robin_hash& operator=(const robin_hash& other) {
    if (&other != this) {
      Hash::operator=(other);
      KeyEqual::operator=(other);
      GrowthPolicy::operator=(other);

      m_buckets_data = other.m_buckets_data;
      m_buckets = m_buckets_data.empty() ? static_empty_bucket_ptr()
                                         : m_buckets_data.data();
      m_bucket_count = other.m_bucket_count;
      m_nb_elements = other.m_nb_elements;

      m_load_threshold = other.m_load_threshold;
      m_min_load_factor = other.m_min_load_factor;
      m_max_load_factor = other.m_max_load_factor;

      m_grow_on_next_insert = other.m_grow_on_next_insert;
      m_try_shrink_on_next_insert = other.m_try_shrink_on_next_insert;
    }

    return *this;
  }

  robin_hash& operator=(robin_hash&& other) {
    other.swap(*this);
    other.clear_and_shrink();

    return *this;
  }

  allocator_type get_allocator() const {
    return m_buckets_data.get_allocator();
  }

  /*
   * Iterators
   */
  iterator begin() noexcept {
    std::size_t i = 0;
    while (i < m_bucket_count && m_buckets[i].empty()) {
      i++;
    }

    return iterator(m_buckets + i);
  }

  const_iterator begin() const noexcept { return cbegin(); }

  const_iterator cbegin() const noexcept {
    std::size_t i = 0;
    while (i < m_bucket_count && m_buckets[i].empty()) {
      i++;
    }

    return const_iterator(m_buckets + i);
  }

  iterator end() noexcept { return iterator(m_buckets + m_bucket_count); }

  const_iterator end() const noexcept { return cend(); }

  const_iterator cend() const noexcept {
    return const_iterator(m_buckets + m_bucket_count);
  }

  /*
   * Capacity
   */
  bool empty() const noexcept { return m_nb_elements == 0; }

  size_type size() const noexcept { return m_nb_elements; }

  size_type max_size() const noexcept { return m_buckets_data.max_size(); }

  /*
   * Modifiers
   */
  void clear() {
    try {
      // 哈希表状态一致性检查
      tsl_rh_safe_assert(m_bucket_count == 0 || m_buckets != nullptr, "Hash table has non-zero buckets but null buckets pointer");
      tsl_rh_safe_assert(m_bucket_count == 0 || m_buckets_data.data() != nullptr, "Hash table has non-zero buckets but null buckets data pointer");
      
      if (m_min_load_factor > 0.0f) {
        clear_and_shrink();
      } else if (m_bucket_count > 0) {
        // 验证桶数据指针有效性
        tsl_rh_safe_assert(m_buckets_data.data() != nullptr, "Attempt to access null buckets data pointer");
        
        // 清除所有桶
        for (std::size_t i = 0; i < m_bucket_count; ++i) {
          // 边界检查
          tsl_rh_safe_assert(i < m_bucket_count, "Invalid bucket index during clear");
          m_buckets_data[i].clear();
        }

        // 更新状态
        m_nb_elements = 0;
        m_grow_on_next_insert = false;
        m_try_shrink_on_next_insert = false;
      }
      
      // 最终状态验证
      tsl_rh_safe_assert(m_nb_elements == 0, "Clear operation failed: element count not zero");
    } catch (...) {
      TSL_RH_THROW_OR_TERMINATE(std::runtime_error, "Unexpected error during clear operation");
    }
  }

  template <typename P>
  std::pair<iterator, bool> insert(P&& value) {
    return insert_impl(KeySelect()(value), std::forward<P>(value));
  }

  template <typename P>
  iterator insert_hint(const_iterator hint, P&& value) {
    if (hint != cend() &&
        compare_keys(KeySelect()(*hint), KeySelect()(value))) {
      return mutable_iterator(hint);
    }

    return insert(std::forward<P>(value)).first;
  }

  template <class InputIt>
  void insert(InputIt first, InputIt last) {
    if (std::is_base_of<
            std::forward_iterator_tag,
            typename std::iterator_traits<InputIt>::iterator_category>::value) {
      const auto nb_elements_insert = std::distance(first, last);
      const size_type nb_free_buckets = m_load_threshold - size();
      tsl_rh_assert(m_load_threshold >= size());

      if (nb_elements_insert > 0 &&
          nb_free_buckets < size_type(nb_elements_insert)) {
        reserve(size() + size_type(nb_elements_insert));
      }
    }

    for (; first != last; ++first) {
      insert(*first);
    }
  }

  template <class K, class M>
  std::pair<iterator, bool> insert_or_assign(K&& key, M&& obj) {
    auto it = try_emplace(std::forward<K>(key), std::forward<M>(obj));
    if (!it.second) {
      it.first.value() = std::forward<M>(obj);
    }

    return it;
  }

  template <class K, class M>
  iterator insert_or_assign(const_iterator hint, K&& key, M&& obj) {
    if (hint != cend() && compare_keys(KeySelect()(*hint), key)) {
      auto it = mutable_iterator(hint);
      it.value() = std::forward<M>(obj);

      return it;
    }

    return insert_or_assign(std::forward<K>(key), std::forward<M>(obj)).first;
  }

  template <class... Args>
  std::pair<iterator, bool> emplace(Args&&... args) {
    return insert(value_type(std::forward<Args>(args)...));
  }

  template <class... Args>
  iterator emplace_hint(const_iterator hint, Args&&... args) {
    return insert_hint(hint, value_type(std::forward<Args>(args)...));
  }

  template <class K, class... Args>
  std::pair<iterator, bool> try_emplace(K&& key, Args&&... args) {
    return insert_impl(key, std::piecewise_construct,
                       std::forward_as_tuple(std::forward<K>(key)),
                       std::forward_as_tuple(std::forward<Args>(args)...));
  }

  template <class K, class... Args>
  iterator try_emplace_hint(const_iterator hint, K&& key, Args&&... args) {
    if (hint != cend() && compare_keys(KeySelect()(*hint), key)) {
      return mutable_iterator(hint);
    }

    return try_emplace(std::forward<K>(key), std::forward<Args>(args)...).first;
  }

  void erase_fast(iterator pos) {
    erase_from_bucket(pos);
  }

  /**
   * Here to avoid `template<class K> size_type erase(const K& key)` being used
   * when we use an `iterator` instead of a `const_iterator`.
   */
  iterator erase(iterator pos) {
    erase_from_bucket(pos);

    /**
     * Erase bucket used a backward shift after clearing the bucket.
     * Check if there is a new value in the bucket, if not get the next
     * non-empty.
     */
    if (pos.m_bucket->empty()) {
      ++pos;
    }

    return pos;
  }

  iterator erase(const_iterator pos) { return erase(mutable_iterator(pos)); }

  iterator erase(const_iterator first, const_iterator last) {
    if (first == last) {
      return mutable_iterator(first);
    }

    auto first_mutable = mutable_iterator(first);
    auto last_mutable = mutable_iterator(last);
    for (auto it = first_mutable.m_bucket; it != last_mutable.m_bucket; ++it) {
      if (!it->empty()) {
        it->clear();
        m_nb_elements--;
      }
    }

    if (last_mutable == end()) {
      m_try_shrink_on_next_insert = true;
      return end();
    }

    /*
     * Backward shift on the values which come after the deleted values.
     * We try to move the values closer to their ideal bucket.
     */
    std::size_t icloser_bucket =
        static_cast<std::size_t>(first_mutable.m_bucket - m_buckets);
    std::size_t ito_move_closer_value =
        static_cast<std::size_t>(last_mutable.m_bucket - m_buckets);
    tsl_rh_assert(ito_move_closer_value > icloser_bucket);

    const std::size_t ireturn_bucket =
        ito_move_closer_value -
        std::min(
            ito_move_closer_value - icloser_bucket,
            std::size_t(
                m_buckets[ito_move_closer_value].dist_from_ideal_bucket()));

    while (ito_move_closer_value < m_bucket_count &&
           m_buckets[ito_move_closer_value].dist_from_ideal_bucket() > 0) {
      icloser_bucket =
          ito_move_closer_value -
          std::min(
              ito_move_closer_value - icloser_bucket,
              std::size_t(
                  m_buckets[ito_move_closer_value].dist_from_ideal_bucket()));

      tsl_rh_assert(m_buckets[icloser_bucket].empty());
      const distance_type new_distance = distance_type(
          m_buckets[ito_move_closer_value].dist_from_ideal_bucket() -
          (ito_move_closer_value - icloser_bucket));
      m_buckets[icloser_bucket].set_value_of_empty_bucket(
          new_distance, m_buckets[ito_move_closer_value].truncated_hash(),
          std::move(m_buckets[ito_move_closer_value].value()));
      m_buckets[ito_move_closer_value].clear();

      ++icloser_bucket;
      ++ito_move_closer_value;
    }

    m_try_shrink_on_next_insert = true;

    return iterator(m_buckets + ireturn_bucket);
  }

  template <class K>
  size_type erase(const K& key) {
    return erase(key, hash_key(key));
  }

  template <class K>
  size_type erase(const K& key, std::size_t hash) {
    auto it = find(key, hash);
    if (it != end()) {
      erase_from_bucket(it);
      return 1;
    } else {
      return 0;
    }
  }

  void swap(robin_hash& other) {
    using std::swap;

    swap(static_cast<Hash&>(*this), static_cast<Hash&>(other));
    swap(static_cast<KeyEqual&>(*this), static_cast<KeyEqual&>(other));
    swap(static_cast<GrowthPolicy&>(*this), static_cast<GrowthPolicy&>(other));
    swap(m_buckets_data, other.m_buckets_data);
    swap(m_buckets, other.m_buckets);
    swap(m_bucket_count, other.m_bucket_count);
    swap(m_nb_elements, other.m_nb_elements);
    swap(m_load_threshold, other.m_load_threshold);
    swap(m_min_load_factor, other.m_min_load_factor);
    swap(m_max_load_factor, other.m_max_load_factor);
    swap(m_grow_on_next_insert, other.m_grow_on_next_insert);
    swap(m_try_shrink_on_next_insert, other.m_try_shrink_on_next_insert);
  }

  /*
   * Lookup
   */
  template <class K, class U = ValueSelect,
            typename std::enable_if<has_mapped_type<U>::value>::type* = nullptr>
  typename U::value_type& at(const K& key) {
    return at(key, hash_key(key));
  }

  template <class K, class U = ValueSelect,
            typename std::enable_if<has_mapped_type<U>::value>::type* = nullptr>
  typename U::value_type& at(const K& key, std::size_t hash) {
    return const_cast<typename U::value_type&>(
        static_cast<const robin_hash*>(this)->at(key, hash));
  }

  template <class K, class U = ValueSelect,
            typename std::enable_if<has_mapped_type<U>::value>::type* = nullptr>
  const typename U::value_type& at(const K& key) const {
    return at(key, hash_key(key));
  }

  template <class K, class U = ValueSelect,
            typename std::enable_if<has_mapped_type<U>::value>::type* = nullptr>
  const typename U::value_type& at(const K& key, std::size_t hash) const {
    auto it = find(key, hash);
    if (it != cend()) {
      return it.value();
    } else {
      TSL_RH_THROW_OR_TERMINATE(std::out_of_range, "Couldn't find key.");
    }
  }

  template <class K, class U = ValueSelect,
            typename std::enable_if<has_mapped_type<U>::value>::type* = nullptr>
  typename U::value_type& operator[](K&& key) {
    return try_emplace(std::forward<K>(key)).first.value();
  }

  template <class K>
  size_type count(const K& key) const {
    return count(key, hash_key(key));
  }

  template <class K>
  size_type count(const K& key, std::size_t hash) const {
    if (find(key, hash) != cend()) {
      return 1;
    } else {
      return 0;
    }
  }

  template <class K>
  iterator find(const K& key) {
    return find_impl(key, hash_key(key));
  }

  template <class K>
  iterator find(const K& key, std::size_t hash) {
    return find_impl(key, hash);
  }

  template <class K>
  const_iterator find(const K& key) const {
    return find_impl(key, hash_key(key));
  }

  template <class K>
  const_iterator find(const K& key, std::size_t hash) const {
    return find_impl(key, hash);
  }

  template <class K>
  bool contains(const K& key) const {
    return contains(key, hash_key(key));
  }

  template <class K>
  bool contains(const K& key, std::size_t hash) const {
    return count(key, hash) != 0;
  }

  template <class K>
  std::pair<iterator, iterator> equal_range(const K& key) {
    return equal_range(key, hash_key(key));
  }

  template <class K>
  std::pair<iterator, iterator> equal_range(const K& key, std::size_t hash) {
    iterator it = find(key, hash);
    return std::make_pair(it, (it == end()) ? it : std::next(it));
  }

  template <class K>
  std::pair<const_iterator, const_iterator> equal_range(const K& key) const {
    return equal_range(key, hash_key(key));
  }

  template <class K>
  std::pair<const_iterator, const_iterator> equal_range(
      const K& key, std::size_t hash) const {
    const_iterator it = find(key, hash);
    return std::make_pair(it, (it == cend()) ? it : std::next(it));
  }

  /*
   * Bucket interface
   */
  size_type bucket_count() const { return m_bucket_count; }

  size_type max_bucket_count() const {
    return std::min(GrowthPolicy::max_bucket_count(),
                    m_buckets_data.max_size());
  }

  /*
   * Hash policy
   */
  float load_factor() const {
    if (bucket_count() == 0) {
      return 0;
    }

    return float(m_nb_elements) / float(bucket_count());
  }

  float min_load_factor() const { return m_min_load_factor; }

  float max_load_factor() const { return m_max_load_factor; }

  void min_load_factor(float ml) {
    m_min_load_factor = std::clamp(ml, float(MINIMUM_MIN_LOAD_FACTOR),
                                   float(MAXIMUM_MIN_LOAD_FACTOR));
  }

  void max_load_factor(float ml) {
    m_max_load_factor = std::clamp(ml, float(MINIMUM_MAX_LOAD_FACTOR),
                                   float(MAXIMUM_MAX_LOAD_FACTOR));
    m_load_threshold = size_type(float(bucket_count()) * m_max_load_factor);
    tsl_rh_assert(bucket_count() == 0 || m_load_threshold < bucket_count());
  }

  void rehash(size_type count_) {
    try {
      // 输入参数有效性检查
      tsl_rh_safe_assert(count_ >= 0, "Invalid rehash count: must be non-negative");
      tsl_rh_safe_assert(count_ <= max_size(), "Rehash count exceeds maximum allowed size");
      
      // 负载因子有效性检查
      const float current_max_load = max_load_factor();
      tsl_rh_safe_assert(current_max_load > 0.0f && current_max_load <= 1.0f, "Invalid max_load_factor value");
      
      // 计算最小需要的桶数量以容纳现有元素
      const size_type min_required_buckets = size_type(std::ceil(float(size()) / current_max_load));
      
      // 确定最终的桶数量
      const size_type final_bucket_count = std::max(count_, min_required_buckets);
      
      // 验证最终桶数量
      tsl_rh_safe_assert(final_bucket_count >= min_required_buckets, "Final bucket count is less than minimum required");
      tsl_rh_safe_assert(final_bucket_count <= max_size(), "Final bucket count exceeds maximum allowed size");
      
      // 只有当桶数量发生变化时才执行rehash
      if (final_bucket_count != m_bucket_count) {
        rehash_impl(final_bucket_count);
        
        // 验证rehash结果
        tsl_rh_safe_assert(m_bucket_count == final_bucket_count, "Rehash failed: bucket count not updated correctly");
        tsl_rh_safe_assert(m_buckets != nullptr || m_bucket_count == 0, "Rehash failed: buckets pointer is null");
      }
    } catch (...) {
      TSL_RH_THROW_OR_TERMINATE(std::runtime_error, "Unexpected error during rehash operation");
    }
  }

  void reserve(size_type count_) {
    try {
      // 输入参数有效性检查
      tsl_rh_safe_assert(count_ >= 0, "Invalid reserve count: must be non-negative");
      tsl_rh_safe_assert(count_ <= max_size(), "Reserve count exceeds maximum allowed size");
      
      // 负载因子有效性检查
      const float current_max_load = max_load_factor();
      tsl_rh_safe_assert(current_max_load > 0.0f && current_max_load <= 1.0f, "Invalid max_load_factor value");
      
      // 计算所需的桶数量
      const size_type required_buckets = size_type(std::ceil(float(count_) / current_max_load));
      
      // 验证计算结果
      tsl_rh_safe_assert(required_buckets >= count_, "Calculated required buckets is less than requested count");
      tsl_rh_safe_assert(required_buckets <= max_size(), "Calculated required buckets exceeds maximum allowed size");
      
      // 执行rehash
      rehash(required_buckets);
      
      // 验证结果
      tsl_rh_safe_assert(bucket_count() >= required_buckets, "Reserve failed: bucket count not sufficient");
    } catch (...) {
      TSL_RH_THROW_OR_TERMINATE(std::runtime_error, "Unexpected error during reserve operation");
    }
  }

  /*
   * Observers
   */
  hasher hash_function() const { return static_cast<const Hash&>(*this); }

  key_equal key_eq() const { return static_cast<const KeyEqual&>(*this); }

  /*
   * Other
   */
  iterator mutable_iterator(const_iterator pos) {
    return iterator(const_cast<bucket_entry*>(pos.m_bucket));
  }

  template <class Serializer>
  void serialize(Serializer& serializer) const {
    serialize_impl(serializer);
  }

  template <class Deserializer>
  void deserialize(Deserializer& deserializer, bool hash_compatible) {
    deserialize_impl(deserializer, hash_compatible);
  }

 private:
  template <class K>
  std::size_t hash_key(const K& key) const {
    return Hash::operator()(key);
  }

  template <class K1, class K2>
  bool compare_keys(const K1& key1, const K2& key2) const {
    return KeyEqual::operator()(key1, key2);
  }

  std::size_t bucket_for_hash(std::size_t hash) const {
    try {
      const std::size_t bucket = GrowthPolicy::bucket_for_hash(hash);
      
      // 桶索引有效性检查
      tsl_rh_safe_assert(bucket < m_bucket_count || (bucket == 0 && m_bucket_count == 0), 
                        "Invalid bucket index calculated from hash");
      tsl_rh_safe_assert(m_bucket_count == 0 || bucket < m_bucket_count, 
                        "Bucket index out of range");
      
      return bucket;
    } catch (...) {
      TSL_RH_THROW_OR_TERMINATE(std::runtime_error, "Unexpected error during bucket calculation");
    }
  }

  template <class U = GrowthPolicy,
            typename std::enable_if<is_power_of_two_policy<U>::value>::type* = nullptr>
  std::size_t next_bucket(std::size_t index) const {
    try {
      // 输入索引有效性检查
      tsl_rh_safe_assert(index < bucket_count(), "Invalid bucket index: out of range");
      tsl_rh_safe_assert(bucket_count() > 0, "Cannot get next bucket from empty hash table");
      
      const std::size_t next_index = (index + 1) & this->m_mask;
      
      // 输出索引有效性检查
      tsl_rh_safe_assert(next_index < bucket_count(), "Invalid next bucket index: out of range");
      
      return next_index;
    } catch (...) {
      TSL_RH_THROW_OR_TERMINATE(std::runtime_error, "Unexpected error during next bucket calculation");
    }
  }

  template <class U = GrowthPolicy,
            typename std::enable_if<!is_power_of_two_policy<U>::value>::type* = nullptr>
  std::size_t next_bucket(std::size_t index) const {
    try {
      // 输入索引有效性检查
      tsl_rh_safe_assert(index < bucket_count(), "Invalid bucket index: out of range");
      tsl_rh_safe_assert(bucket_count() > 0, "Cannot get next bucket from empty hash table");
      
      const std::size_t next_index = (index + 1) % bucket_count();
      
      // 输出索引有效性检查
      tsl_rh_safe_assert(next_index < bucket_count(), "Invalid next bucket index: out of range");
      
      return next_index;
    } catch (...) {
      TSL_RH_THROW_OR_TERMINATE(std::runtime_error, "Unexpected error during next bucket calculation");
    }
  }

  template <class K>
  iterator find_impl(const K& key, std::size_t hash) {
    return mutable_iterator(
        static_cast<const robin_hash*>(this)->find(key, hash));
  }

  template <class K>
  const_iterator find_impl(const K& key, std::size_t hash) const {
    try {
      // 哈希表状态一致性检查
      tsl_rh_safe_assert(m_bucket_count == 0 || m_buckets != nullptr, "Hash table has non-zero buckets but null buckets pointer");
      tsl_rh_safe_assert(m_nb_elements == 0 || m_bucket_count > 0, "Hash table has non-zero elements but zero buckets");
      tsl_rh_safe_assert(m_nb_elements <= m_bucket_count, "Hash table has more elements than buckets");
      
      // 空表快速返回
      if (m_nb_elements == 0) {
        return cend();
      }
      
      // 桶数量检查
      if (m_bucket_count == 0) {
        tsl_rh_safe_assert(m_nb_elements == 0, "Hash table has non-zero elements but zero buckets");
        return cend();
      }
      
      std::size_t ibucket = bucket_for_hash(hash);
      distance_type dist_from_ideal_bucket = 0;

      // 桶索引有效性检查
      tsl_rh_safe_assert(ibucket < m_bucket_count, "Invalid bucket index calculated from hash");
      tsl_rh_safe_assert(m_buckets != nullptr, "Attempt to access null buckets pointer");

      while (dist_from_ideal_bucket <= m_buckets[ibucket].dist_from_ideal_bucket()) {
        // 桶状态一致性检查
        tsl_rh_safe_assert(!m_buckets[ibucket].empty(), "Invalid bucket state: empty bucket with non-negative distance");
        tsl_rh_safe_assert(m_buckets[ibucket].dist_from_ideal_bucket() >= 0, "Invalid bucket distance: negative value");
        tsl_rh_safe_assert(m_buckets[ibucket].dist_from_ideal_bucket() <= bucket_entry::DIST_FROM_IDEAL_BUCKET_LIMIT, "Invalid bucket distance: exceeds maximum limit");
        
        // 键匹配检查
        if (TSL_RH_LIKELY((!USE_STORED_HASH_ON_LOOKUP || m_buckets[ibucket].bucket_hash_equal(hash)) &&
                         compare_keys(KeySelect()(m_buckets[ibucket].value()), key))) {
          return const_iterator(m_buckets + ibucket);
        }

        // 移动到下一个桶
        ibucket = next_bucket(ibucket);
        dist_from_ideal_bucket++;
        
        // 边界检查和无限循环防护
        tsl_rh_safe_assert(ibucket < m_bucket_count, "Invalid bucket index after increment");
        tsl_rh_safe_assert(dist_from_ideal_bucket <= bucket_entry::DIST_FROM_IDEAL_BUCKET_LIMIT, "Excessive distance from ideal bucket during find - potential infinite loop");
        tsl_rh_safe_assert(dist_from_ideal_bucket <= m_bucket_count, "Distance exceeds bucket count - potential infinite loop");
      }

      return cend();
    } catch (...) {
      TSL_RH_THROW_OR_TERMINATE(std::runtime_error, "Unexpected error during find operation");
    }
  }

  void erase_from_bucket(iterator pos) {
    try {
      // 空指针和边界检查
      tsl_rh_safe_assert(pos.m_bucket != nullptr, "Attempt to erase from null iterator");
      tsl_rh_safe_assert(m_buckets != nullptr, "Attempt to erase from hash table with null buckets pointer");
      tsl_rh_safe_assert(m_bucket_count > 0, "Attempt to erase from hash table with zero buckets");
      
      // 检查迭代器是否有效
      std::size_t bucket_index = static_cast<std::size_t>(pos.m_bucket - m_buckets);
      tsl_rh_safe_assert(bucket_index < m_bucket_count, "Invalid bucket index from iterator");
      tsl_rh_safe_assert(!pos.m_bucket->empty(), "Attempt to erase from empty bucket");

      pos.m_bucket->clear();
      m_nb_elements--;

      /**
       * Backward shift, swap the empty bucket, previous_ibucket, with the values
       * on its right, ibucket, until we cross another empty bucket or if the
       * other bucket has a distance_from_ideal_bucket == 0.
       *
       * We try to move the values closer to their ideal bucket.
       */
      std::size_t previous_ibucket = bucket_index;
      std::size_t ibucket = next_bucket(previous_ibucket);

      while (m_buckets[ibucket].dist_from_ideal_bucket() > 0) {
        tsl_rh_safe_assert(m_buckets[previous_ibucket].empty(), "Invalid bucket state: previous bucket should be empty during backward shift");
        tsl_rh_safe_assert(!m_buckets[ibucket].empty(), "Invalid bucket state: current bucket should not be empty during backward shift");

        const distance_type new_distance = 
            distance_type(m_buckets[ibucket].dist_from_ideal_bucket() - 1);
        m_buckets[previous_ibucket].set_value_of_empty_bucket(
            new_distance, m_buckets[ibucket].truncated_hash(),
            std::move(m_buckets[ibucket].value()));
        m_buckets[ibucket].clear();

        previous_ibucket = ibucket;
        ibucket = next_bucket(ibucket);
        
        // 防止无限循环
        tsl_rh_safe_assert(previous_ibucket < m_bucket_count, "Exceeded bucket count during backward shift");
      }
      m_try_shrink_on_next_insert = true;
    } catch (...) {
      TSL_RH_THROW_OR_TERMINATE(std::runtime_error, 
                                "Unexpected error during erase");
    }
  }

  template <class K, class... Args>
  std::pair<iterator, bool> insert_impl(const K& key,
                                        Args&&... value_type_args) {
    try {
      const std::size_t hash = hash_key(key);

      std::size_t ibucket = bucket_for_hash(hash);
      distance_type dist_from_ideal_bucket = 0;

      // 空指针和边界检查
      tsl_rh_safe_assert(m_buckets != nullptr, "Attempt to insert into hash table with null buckets pointer");
      tsl_rh_safe_assert(m_bucket_count > 0 || m_nb_elements == 0, "Hash table has zero buckets but non-zero elements");
      tsl_rh_safe_assert(ibucket < m_bucket_count, "Invalid bucket index calculated from hash");

      while (dist_from_ideal_bucket <=
             m_buckets[ibucket].dist_from_ideal_bucket()) {
        // 空桶检查
        tsl_rh_safe_assert(!m_buckets[ibucket].empty(), "Invalid bucket state: empty bucket with non-negative distance");
        
        if ((!USE_STORED_HASH_ON_LOOKUP ||
             m_buckets[ibucket].bucket_hash_equal(hash)) &&
            compare_keys(KeySelect()(m_buckets[ibucket].value()), key)) {
          return std::make_pair(iterator(m_buckets + ibucket), false);
        }

        ibucket = next_bucket(ibucket);
        dist_from_ideal_bucket++;
        
        // 防止无限循环
        tsl_rh_safe_assert(dist_from_ideal_bucket <= bucket_entry::DIST_FROM_IDEAL_BUCKET_LIMIT, "Excessive distance from ideal bucket during insert");
      }

      while (rehash_on_extreme_load(dist_from_ideal_bucket)) {
        ibucket = bucket_for_hash(hash);
        dist_from_ideal_bucket = 0;

        while (dist_from_ideal_bucket <=
               m_buckets[ibucket].dist_from_ideal_bucket()) {
          ibucket = next_bucket(ibucket);
          dist_from_ideal_bucket++;
        }
      }

      if (m_buckets[ibucket].empty()) {
        m_buckets[ibucket].set_value_of_empty_bucket(
            dist_from_ideal_bucket, bucket_entry::truncate_hash(hash),
            std::forward<Args>(value_type_args)...);
      } else {
        insert_value(ibucket, dist_from_ideal_bucket,
                     bucket_entry::truncate_hash(hash),
                     std::forward<Args>(value_type_args)...);
      }

      m_nb_elements++;
      /*
       * The value will be inserted in ibucket in any case, either because it was
       * empty or by stealing the bucket (robin hood).
       */
      return std::make_pair(iterator(m_buckets + ibucket), true);
    } catch (const std::bad_alloc& e) {
      TSL_RH_THROW_OR_TERMINATE(allocation_error, 
                                "Memory allocation failed during insert: " + std::string(e.what()));
    } catch (...) {
      TSL_RH_THROW_OR_TERMINATE(std::runtime_error, 
                                "Unexpected error during insert");
    }
  }

  template <class... Args>
  void insert_value(std::size_t ibucket, distance_type dist_from_ideal_bucket,
                    truncated_hash_type hash, Args&&... value_type_args) {
    value_type value(std::forward<Args>(value_type_args)...);
    insert_value_impl(ibucket, dist_from_ideal_bucket, hash, value);
  }

  void insert_value(std::size_t ibucket, distance_type dist_from_ideal_bucket,
                    truncated_hash_type hash, value_type&& value) {
    insert_value_impl(ibucket, dist_from_ideal_bucket, hash, value);
  }

  /*
   * We don't use `value_type&& value` as last argument due to a bug in MSVC
   * when `value_type` is a pointer, The compiler is not able to see the
   * difference between `std::string*` and `std::string*&&` resulting in a
   * compilation error.
   *
   * The `value` will be in a moved state at the end of the function.
   */
  void insert_value_impl(std::size_t ibucket,
                         distance_type dist_from_ideal_bucket,
                         truncated_hash_type hash, value_type& value) {
    tsl_rh_assert(dist_from_ideal_bucket >
                  m_buckets[ibucket].dist_from_ideal_bucket());
    m_buckets[ibucket].swap_with_value_in_bucket(dist_from_ideal_bucket, hash,
                                                 value);
    ibucket = next_bucket(ibucket);
    dist_from_ideal_bucket++;

    while (!m_buckets[ibucket].empty()) {
      if (dist_from_ideal_bucket >
          m_buckets[ibucket].dist_from_ideal_bucket()) {
        if (dist_from_ideal_bucket >
            bucket_entry::DIST_FROM_IDEAL_BUCKET_LIMIT) {
          /**
           * The number of probes is really high, rehash the map on the next
           * insert. Difficult to do now as rehash may throw an exception.
           */
          m_grow_on_next_insert = true;
        }

        m_buckets[ibucket].swap_with_value_in_bucket(dist_from_ideal_bucket,
                                                     hash, value);
      }

      ibucket = next_bucket(ibucket);
      dist_from_ideal_bucket++;
    }

    m_buckets[ibucket].set_value_of_empty_bucket(dist_from_ideal_bucket, hash,
                                                 std::move(value));
  }

  void rehash_impl(size_type count_) {
    try {
      robin_hash new_table(count_, static_cast<Hash&>(*this),
                           static_cast<KeyEqual&>(*this), get_allocator(),
                           m_min_load_factor, m_max_load_factor);
      tsl_rh_assert(size() <= new_table.m_load_threshold);

      const bool use_stored_hash = 
          USE_STORED_HASH_ON_REHASH(new_table.bucket_count());
      for (auto& bucket : m_buckets_data) {
        if (bucket.empty()) {
          continue;
        }

        const std::size_t hash = 
            use_stored_hash ? bucket.truncated_hash()
                            : new_table.hash_key(KeySelect()(bucket.value()));

        new_table.insert_value_on_rehash(new_table.bucket_for_hash(hash), 0,
                                         bucket_entry::truncate_hash(hash),
                                         std::move(bucket.value()));
      }

      new_table.m_nb_elements = m_nb_elements;
      new_table.swap(*this);
    } catch (const std::bad_alloc& e) {
      TSL_RH_THROW_OR_TERMINATE(allocation_error, 
                                "Memory allocation failed during rehash: " + std::string(e.what()));
    } catch (...) {
      TSL_RH_THROW_OR_TERMINATE(std::runtime_error, 
                                "Unexpected error during rehash");
    }
  }

  void clear_and_shrink() {
    try {
      // 哈希表状态一致性检查
      tsl_rh_safe_assert(m_bucket_count == 0 || m_buckets != nullptr, "Hash table has non-zero buckets but null buckets pointer");
      tsl_rh_safe_assert(m_bucket_count == 0 || m_buckets_data.data() != nullptr, "Hash table has non-zero buckets but null buckets data pointer");
      
      // 清除增长策略
      GrowthPolicy::clear();
      
      // 清除桶数据
      if (!m_buckets_data.empty()) {
        m_buckets_data.clear();
        
        // 验证桶数据已被清除
        tsl_rh_safe_assert(m_buckets_data.empty(), "Failed to clear buckets data");
      }
      
      // 更新状态指针和计数器
      m_buckets = static_empty_bucket_ptr();
      m_bucket_count = 0;
      m_nb_elements = 0;
      m_load_threshold = 0;
      m_grow_on_next_insert = false;
      m_try_shrink_on_next_insert = false;
      
      // 最终状态验证
      tsl_rh_safe_assert(m_bucket_count == 0, "Clear and shrink failed: bucket count not zero");
      tsl_rh_safe_assert(m_nb_elements == 0, "Clear and shrink failed: element count not zero");
      tsl_rh_safe_assert(m_buckets == static_empty_bucket_ptr(), "Clear and shrink failed: buckets pointer not reset");
      tsl_rh_safe_assert(m_buckets_data.empty(), "Clear and shrink failed: buckets data not empty");
      tsl_rh_safe_assert(m_load_threshold == 0, "Clear and shrink failed: load threshold not reset");
    } catch (...) {
      TSL_RH_THROW_OR_TERMINATE(std::runtime_error, "Unexpected error during clear and shrink operation");
    }
  }

  void insert_value_on_rehash(std::size_t ibucket,
                              distance_type dist_from_ideal_bucket,
                              truncated_hash_type hash, value_type&& value) {
    try {
      // 输入参数有效性检查
      tsl_rh_safe_assert(ibucket < m_bucket_count, "Invalid bucket index: out of range");
      tsl_rh_safe_assert(dist_from_ideal_bucket >= 0, "Invalid distance: must be non-negative");
      tsl_rh_safe_assert(m_buckets != nullptr, "Attempt to access null buckets pointer");
      tsl_rh_safe_assert(m_bucket_count > 0, "Cannot insert into empty hash table");
      
      // 防止无限循环的保护
      const std::size_t max_iterations = m_bucket_count * 2; // 安全限制
      std::size_t iterations = 0;
      
      while (true) {
        // 迭代次数检查，防止无限循环
        tsl_rh_safe_assert(iterations < max_iterations, "Insert on rehash failed: infinite loop detected");
        iterations++;
        
        // 桶索引有效性检查
        tsl_rh_safe_assert(ibucket < m_bucket_count, "Invalid bucket index during insert on rehash");
        
        // 距离有效性检查
        tsl_rh_safe_assert(dist_from_ideal_bucket >= 0, "Invalid distance during insert on rehash");
        tsl_rh_safe_assert(dist_from_ideal_bucket <= bucket_entry::DIST_FROM_IDEAL_BUCKET_LIMIT, 
                          "Distance exceeds maximum allowed limit");
        
        if (dist_from_ideal_bucket > m_buckets[ibucket].dist_from_ideal_bucket()) {
          if (m_buckets[ibucket].empty()) {
            // 插入到空桶
            m_buckets[ibucket].set_value_of_empty_bucket(dist_from_ideal_bucket, hash, std::move(value));
            return;
          } else {
            // 与现有桶交换
            m_buckets[ibucket].swap_with_value_in_bucket(dist_from_ideal_bucket, hash, value);
          }
        }

        dist_from_ideal_bucket++;
        ibucket = next_bucket(ibucket);
      }
    } catch (...) {
      TSL_RH_THROW_OR_TERMINATE(std::runtime_error, "Unexpected error during insert on rehash operation");
    }
  }

  /**
   * Grow the table if m_grow_on_next_insert is true or we reached the
   * max_load_factor. Shrink the table if m_try_shrink_on_next_insert is true
   * (an erase occurred) and we're below the min_load_factor.
   *
   * Return true if the table has been rehashed.
   */
  bool rehash_on_extreme_load(distance_type curr_dist_from_ideal_bucket) {
    try {
      // 输入参数有效性检查
      tsl_rh_safe_assert(curr_dist_from_ideal_bucket >= 0, "Invalid distance: must be non-negative");
      
      // 哈希表状态一致性检查
      tsl_rh_safe_assert(m_bucket_count == 0 || m_buckets != nullptr, "Hash table has non-zero buckets but null buckets pointer");
      tsl_rh_safe_assert(m_nb_elements <= m_bucket_count, "Invalid state: more elements than buckets");
      
      // 检查是否需要扩容
      if (m_grow_on_next_insert ||
          curr_dist_from_ideal_bucket > bucket_entry::DIST_FROM_IDEAL_BUCKET_LIMIT ||
          size() >= m_load_threshold) {
        
        // 计算新的桶数量
        const std::size_t next_bucket_count = GrowthPolicy::next_bucket_count();
        tsl_rh_safe_assert(next_bucket_count > m_bucket_count, "Next bucket count must be larger than current");
        
        // 执行扩容
        rehash_impl(next_bucket_count);
        m_grow_on_next_insert = false;

        // 验证扩容结果
        tsl_rh_safe_assert(m_bucket_count == next_bucket_count, "Rehash failed: bucket count not updated correctly");
        tsl_rh_safe_assert(m_buckets != nullptr, "Rehash failed: buckets pointer is null");
        
        return true;
      }

      // 检查是否需要缩容
      if (m_try_shrink_on_next_insert) {
        m_try_shrink_on_next_insert = false;
        
        // 验证负载因子有效性
        tsl_rh_safe_assert(m_min_load_factor >= 0.0f && m_min_load_factor <= 1.0f, "Invalid min_load_factor value");
        
        if (m_min_load_factor != 0.0f && load_factor() < m_min_load_factor) {
          // 计算新的容量
          const std::size_t new_capacity = size() + 1;
          tsl_rh_safe_assert(new_capacity <= m_bucket_count, "New capacity must be less than or equal to current bucket count");
          
          // 执行缩容
          reserve(new_capacity);

          // 验证缩容结果
          tsl_rh_safe_assert(m_bucket_count >= new_capacity, "Reserve failed: bucket count not sufficient");
          
          return true;
        }
      }

      return false;
    } catch (...) {
      TSL_RH_THROW_OR_TERMINATE(std::runtime_error, "Unexpected error during rehash on extreme load operation");
    }
  }

  template <class Serializer>
  void serialize_impl(Serializer& serializer) const {
    try {
      // 空指针检查
      tsl_rh_safe_assert(m_buckets != nullptr || m_bucket_count == 0, "Attempt to serialize hash table with null buckets pointer");
      
      // 数据完整性检查
      tsl_rh_safe_assert(m_nb_elements <= m_bucket_count, "Invalid state: more elements than buckets");
      tsl_rh_safe_assert(m_min_load_factor >= MINIMUM_MIN_LOAD_FACTOR && m_min_load_factor <= MAXIMUM_MIN_LOAD_FACTOR, "Invalid min_load_factor value");
      tsl_rh_safe_assert(m_max_load_factor >= MINIMUM_MAX_LOAD_FACTOR && m_max_load_factor <= MAXIMUM_MAX_LOAD_FACTOR, "Invalid max_load_factor value");
      tsl_rh_safe_assert(m_bucket_count == 0 || m_buckets_data.data() != nullptr, "Hash table has non-zero buckets but null buckets data pointer");
      
      // 序列化协议版本
      const slz_size_type version = SERIALIZATION_PROTOCOL_VERSION;
      serializer(version);

    // Indicate if the truncated hash of each bucket is stored. Use a
    // std::int16_t instead of a bool to avoid the need for the serializer to
    // support an extra 'bool' type.
    const std::int16_t hash_stored_for_bucket = 
        static_cast<std::int16_t>(STORE_HASH);
    serializer(hash_stored_for_bucket);

    const slz_size_type nb_elements = m_nb_elements;
    serializer(nb_elements);

    const slz_size_type bucket_count = m_buckets_data.size();
    serializer(bucket_count);

    const float min_load_factor = m_min_load_factor;
    serializer(min_load_factor);

    const float max_load_factor = m_max_load_factor;
    serializer(max_load_factor);

    // 计算并序列化校验和
    std::uint32_t checksum = 0;
    for (const bucket_entry& bucket : m_buckets_data) {
      if (!bucket.empty()) {
        // 简单的校验和计算，实际项目中可以使用更复杂的算法
        checksum ^= static_cast<std::uint32_t>(bucket.dist_from_ideal_bucket());
        if (STORE_HASH) {
          checksum ^= bucket.truncated_hash();
        }
      }
    }
    serializer(checksum);

    for (const bucket_entry& bucket : m_buckets_data) {
      if (bucket.empty()) {
        const std::int16_t empty_bucket = 
            bucket_entry::EMPTY_MARKER_DIST_FROM_IDEAL_BUCKET;
        serializer(empty_bucket);
      } else {
        const std::int16_t dist_from_ideal_bucket = 
            bucket.dist_from_ideal_bucket();
        
        // 边界检查
        tsl_rh_safe_assert(dist_from_ideal_bucket >= 0 && dist_from_ideal_bucket <= bucket_entry::DIST_FROM_IDEAL_BUCKET_LIMIT, "Invalid distance from ideal bucket");
        
        serializer(dist_from_ideal_bucket);
        if (STORE_HASH) {
          const std::uint32_t truncated_hash = bucket.truncated_hash();
          serializer(truncated_hash);
        }
        serializer(bucket.value());
      }
    }
    } catch (...) {
      TSL_RH_THROW_OR_TERMINATE(std::runtime_error, "Unexpected error during serialization operation");
    }
  }

  template <class Deserializer>
  void deserialize_impl(Deserializer& deserializer, bool hash_compatible) {
    try {
      tsl_rh_assert(m_buckets_data.empty());  // Current hash table must be empty
      
      // 空指针安全检查
      tsl_rh_safe_assert(&deserializer != nullptr, "Deserializer pointer is null");
      
      // 验证当前哈希表为空
      tsl_rh_safe_assert(m_nb_elements == 0, "Cannot deserialize into non-empty hash table");
      tsl_rh_safe_assert(m_bucket_count == 0, "Cannot deserialize into non-empty hash table");
      
      const slz_size_type version = 
          deserialize_value<slz_size_type>(deserializer);
      // For now we only have one version of the serialization protocol.
      // If it doesn't match there is a problem with the file.
      if (version != SERIALIZATION_PROTOCOL_VERSION) {
        TSL_RH_THROW_OR_TERMINATE(std::runtime_error,
                                  "Can't deserialize the ordered_map/set. "
                                  "The protocol version header is invalid.");
      }

    const bool hash_stored_for_bucket = 
        deserialize_value<std::int16_t>(deserializer) ? true : false;
    if (hash_compatible && STORE_HASH != hash_stored_for_bucket) {
      TSL_RH_THROW_OR_TERMINATE(
          std::runtime_error,
          "Can't deserialize a map with a different StoreHash "
          "than the one used during the serialization when "
          "hash compatibility is used");
    }

    const slz_size_type nb_elements = 
        deserialize_value<slz_size_type>(deserializer);
    const slz_size_type bucket_count_ds = 
        deserialize_value<slz_size_type>(deserializer);
    const float min_load_factor = deserialize_value<float>(deserializer);
    const float max_load_factor = deserialize_value<float>(deserializer);

    // 数据完整性校验：元素数量不能超过桶数量
    tsl_rh_safe_assert(nb_elements <= bucket_count_ds || bucket_count_ds == 0, 
                      "Number of elements exceeds bucket count");
    
    // 桶数量边界检查
    tsl_rh_safe_assert(bucket_count_ds <= max_size(), 
                      "Deserialized bucket count exceeds maximum allowed size");

    if (min_load_factor < MINIMUM_MIN_LOAD_FACTOR ||
        min_load_factor > MAXIMUM_MIN_LOAD_FACTOR) {
      TSL_RH_THROW_OR_TERMINATE(
          std::runtime_error,
          "Invalid min_load_factor. Check that the serializer "
          "and deserializer support floats correctly as they "
          "can be converted implicitly to ints.");
    }

    if (max_load_factor < MINIMUM_MAX_LOAD_FACTOR ||
        max_load_factor > MAXIMUM_MAX_LOAD_FACTOR) {
      TSL_RH_THROW_OR_TERMINATE(
          std::runtime_error,
          "Invalid max_load_factor. Check that the serializer "
          "and deserializer support floats correctly as they "
          "can be converted implicitly to ints.");
    }

    this->min_load_factor(min_load_factor);
    this->max_load_factor(max_load_factor);

    if (bucket_count_ds == 0) {
      tsl_rh_assert(nb_elements == 0);
      return;
    }
    
    // 负载因子合理性检查
    tsl_rh_safe_assert(max_load_factor > 0.0f && max_load_factor <= 1.0f, 
                      "Invalid max_load_factor value");
    tsl_rh_safe_assert(min_load_factor >= 0.0f && min_load_factor < max_load_factor, 
                      "Invalid min_load_factor value relative to max_load_factor");

    if (!hash_compatible) {
      reserve(numeric_cast<size_type>(nb_elements,
                                      "Deserialized nb_elements is too big."));
      for (slz_size_type ibucket = 0; ibucket < bucket_count_ds; ibucket++) {
        const distance_type dist_from_ideal_bucket = 
            deserialize_value<std::int16_t>(deserializer);
            
        // 距离边界检查
        tsl_rh_safe_assert(dist_from_ideal_bucket == bucket_entry::EMPTY_MARKER_DIST_FROM_IDEAL_BUCKET || 
                          (dist_from_ideal_bucket >= 0 && dist_from_ideal_bucket < static_cast<distance_type>(bucket_count_ds)), 
                          "Invalid distance from ideal bucket");
                          
        if (dist_from_ideal_bucket != 
            bucket_entry::EMPTY_MARKER_DIST_FROM_IDEAL_BUCKET) {
          if (hash_stored_for_bucket) {
            const std::uint32_t hash = deserialize_value<std::uint32_t>(deserializer);
            TSL_RH_UNUSED(hash);
          }

          insert(deserialize_value<value_type>(deserializer));
        }
      }

      tsl_rh_assert(nb_elements == size());
    } else {
      m_bucket_count = numeric_cast<size_type>(
          bucket_count_ds, "Deserialized bucket_count is too big.");
      // Recompute m_load_threshold, during max_load_factor() the bucket count
      // was still 0 which would trigger rehash on first insert
      m_load_threshold = size_type(float(bucket_count()) * m_max_load_factor);

      GrowthPolicy::operator=(GrowthPolicy(m_bucket_count));
      // GrowthPolicy should not modify the bucket count we got from
      // deserialization
      if (m_bucket_count != bucket_count_ds) {
        TSL_RH_THROW_OR_TERMINATE(std::runtime_error,
                                  "The GrowthPolicy is not the same even "
                                  "though hash_compatible is true.");
      }

      m_nb_elements = numeric_cast<size_type>(
          nb_elements, "Deserialized nb_elements is too big.");
      m_buckets_data.resize(m_bucket_count);
      m_buckets = m_buckets_data.data();

      for (bucket_entry& bucket : m_buckets_data) {
        const distance_type dist_from_ideal_bucket = 
            deserialize_value<std::int16_t>(deserializer);
            
        // 距离边界检查
        tsl_rh_safe_assert(dist_from_ideal_bucket == bucket_entry::EMPTY_MARKER_DIST_FROM_IDEAL_BUCKET || 
                          (dist_from_ideal_bucket >= 0 && dist_from_ideal_bucket < static_cast<distance_type>(m_bucket_count)), 
                          "Invalid distance from ideal bucket");
                          
        if (dist_from_ideal_bucket != 
            bucket_entry::EMPTY_MARKER_DIST_FROM_IDEAL_BUCKET) {
          truncated_hash_type truncated_hash = 0;
          if (hash_stored_for_bucket) {
            tsl_rh_assert(hash_stored_for_bucket);
            truncated_hash = deserialize_value<std::uint32_t>(deserializer);
          }

          bucket.set_value_of_empty_bucket(
              dist_from_ideal_bucket, truncated_hash,
              deserialize_value<value_type>(deserializer));
        }
      }

      if (!m_buckets_data.empty()) {
        m_buckets_data.back().set_as_last_bucket();
      }
    }
    } catch (...) {
      TSL_RH_THROW_OR_TERMINATE(std::runtime_error, "Unexpected error during deserialization operation");
    }
  }

 public:
  static const size_type DEFAULT_INIT_BUCKETS_SIZE = 0;

  static constexpr float DEFAULT_MAX_LOAD_FACTOR = 0.5f;
  static constexpr float MINIMUM_MAX_LOAD_FACTOR = 0.2f;
  static constexpr float MAXIMUM_MAX_LOAD_FACTOR = 0.95f;

  static constexpr float DEFAULT_MIN_LOAD_FACTOR = 0.0f;
  static constexpr float MINIMUM_MIN_LOAD_FACTOR = 0.0f;
  static constexpr float MAXIMUM_MIN_LOAD_FACTOR = 0.15f;

  static_assert(MINIMUM_MAX_LOAD_FACTOR < MAXIMUM_MAX_LOAD_FACTOR,
                "MINIMUM_MAX_LOAD_FACTOR should be < MAXIMUM_MAX_LOAD_FACTOR");
  static_assert(MINIMUM_MIN_LOAD_FACTOR < MAXIMUM_MIN_LOAD_FACTOR,
                "MINIMUM_MIN_LOAD_FACTOR should be < MAXIMUM_MIN_LOAD_FACTOR");
  static_assert(MAXIMUM_MIN_LOAD_FACTOR < MINIMUM_MAX_LOAD_FACTOR,
                "MAXIMUM_MIN_LOAD_FACTOR should be < MINIMUM_MAX_LOAD_FACTOR");

 private:
  /**
   * Protocol version currenlty used for serialization.
   */
  static const slz_size_type SERIALIZATION_PROTOCOL_VERSION = 1;

  /**
   * Return an always valid pointer to an static empty bucket_entry with
   * last_bucket() == true.
   */
  bucket_entry* static_empty_bucket_ptr() noexcept {
    static bucket_entry empty_bucket(true);
    tsl_rh_assert(empty_bucket.empty());
    return &empty_bucket;
  }

 private:
  buckets_container_type m_buckets_data;

  /**
   * Points to m_buckets_data.data() if !m_buckets_data.empty() otherwise points
   * to static_empty_bucket_ptr. This variable is useful to avoid the cost of
   * checking if m_buckets_data is empty when trying to find an element.
   *
   * TODO Remove m_buckets_data and only use a pointer instead of a
   * pointer+vector to save some space in the robin_hash object. Manage the
   * Allocator manually.
   */
  bucket_entry* m_buckets;

  /**
   * Used a lot in find, avoid the call to m_buckets_data.size() which is a bit
   * slower.
   */
  size_type m_bucket_count;

  size_type m_nb_elements;

  size_type m_load_threshold;

  float m_min_load_factor;
  float m_max_load_factor;

  bool m_grow_on_next_insert;

  /**
   * We can't shrink down the map on erase operations as the erase methods need
   * to return the next iterator. Shrinking the map would invalidate all the
   * iterators and we could not return the next iterator in a meaningful way, On
   * erase, we thus just indicate on erase that we should try to shrink the hash
   * table on the next insert if we go below the min_load_factor.
   */
  bool m_try_shrink_on_next_insert;
};

}  // namespace detail_robin_hash

}  // namespace tsl

#endif
