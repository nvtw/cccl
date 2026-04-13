// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_radix_sort.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include <c2h/catch2_test_helper.h>

// ============================================================================
// Thin wrappers to disambiguate indirect vs standard overloads.
// The indirect API takes (const NumItemsT* d_num_items, NumItemsT max_num_items)
// which is ambiguous with the standard API when default args are considered.
// These wrappers explicitly pass begin_bit/end_bit to resolve the ambiguity.
// ============================================================================

template <typename KeyT, typename ValueT, typename NumItemsT>
cudaError_t indirect_sort_pairs(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  const KeyT* d_keys_in,
  KeyT* d_keys_out,
  const ValueT* d_values_in,
  ValueT* d_values_out,
  const NumItemsT* d_num_items,
  NumItemsT max_num_items,
  cudaStream_t stream = 0)
{
  return cub::DeviceRadixSort::SortPairs(
    d_temp_storage,
    temp_storage_bytes,
    d_keys_in,
    d_keys_out,
    d_values_in,
    d_values_out,
    d_num_items,
    max_num_items,
    0,
    static_cast<int>(sizeof(KeyT) * 8),
    stream);
}

template <typename KeyT, typename ValueT, typename NumItemsT>
cudaError_t indirect_sort_pairs_descending(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  const KeyT* d_keys_in,
  KeyT* d_keys_out,
  const ValueT* d_values_in,
  ValueT* d_values_out,
  const NumItemsT* d_num_items,
  NumItemsT max_num_items,
  cudaStream_t stream = 0)
{
  return cub::DeviceRadixSort::SortPairsDescending(
    d_temp_storage,
    temp_storage_bytes,
    d_keys_in,
    d_keys_out,
    d_values_in,
    d_values_out,
    d_num_items,
    max_num_items,
    0,
    static_cast<int>(sizeof(KeyT) * 8),
    stream);
}

template <typename KeyT, typename NumItemsT>
cudaError_t indirect_sort_keys(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  const KeyT* d_keys_in,
  KeyT* d_keys_out,
  const NumItemsT* d_num_items,
  NumItemsT max_num_items,
  cudaStream_t stream = 0)
{
  return cub::DeviceRadixSort::SortKeys(
    d_temp_storage,
    temp_storage_bytes,
    d_keys_in,
    d_keys_out,
    d_num_items,
    max_num_items,
    0,
    static_cast<int>(sizeof(KeyT) * 8),
    stream);
}

template <typename KeyT, typename NumItemsT>
cudaError_t indirect_sort_keys_descending(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  const KeyT* d_keys_in,
  KeyT* d_keys_out,
  const NumItemsT* d_num_items,
  NumItemsT max_num_items,
  cudaStream_t stream = 0)
{
  return cub::DeviceRadixSort::SortKeysDescending(
    d_temp_storage,
    temp_storage_bytes,
    d_keys_in,
    d_keys_out,
    d_num_items,
    max_num_items,
    0,
    static_cast<int>(sizeof(KeyT) * 8),
    stream);
}

// ============================================================================
// Reference implementations
// ============================================================================

template <typename KeyT>
std::vector<KeyT> sort_reference(const std::vector<KeyT>& input, int num_items, bool descending)
{
  std::vector<KeyT> result(input.begin(), input.begin() + num_items);
  if (descending)
  {
    std::sort(result.begin(), result.end(), std::greater<KeyT>{});
  }
  else
  {
    std::sort(result.begin(), result.end());
  }
  return result;
}

template <typename KeyT, typename ValueT>
void sort_pairs_reference(
  const std::vector<KeyT>& keys_in,
  const std::vector<ValueT>& values_in,
  std::vector<KeyT>& keys_out,
  std::vector<ValueT>& values_out,
  int num_items,
  bool descending)
{
  std::vector<std::pair<KeyT, ValueT>> pairs(num_items);
  for (int i = 0; i < num_items; ++i)
  {
    pairs[i] = {keys_in[i], values_in[i]};
  }
  if (descending)
  {
    std::stable_sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) {
      return a.first > b.first;
    });
  }
  else
  {
    std::stable_sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) {
      return a.first < b.first;
    });
  }
  keys_out.resize(num_items);
  values_out.resize(num_items);
  for (int i = 0; i < num_items; ++i)
  {
    keys_out[i]   = pairs[i].first;
    values_out[i] = pairs[i].second;
  }
}

// ============================================================================
// Helper: run indirect sort and verify against reference AND standard CUB sort
// ============================================================================

template <typename KeyT>
void verify_indirect_sort_keys(
  const std::vector<KeyT>& h_keys,
  int max_num_items,
  int actual_num_items,
  bool descending,
  bool compare_with_standard = true)
{
  thrust::device_vector<KeyT> d_keys_in(h_keys.begin(), h_keys.end());
  thrust::device_vector<KeyT> d_keys_out_indirect(max_num_items, KeyT{});
  thrust::device_vector<KeyT> d_keys_out_standard(max_num_items, KeyT{});

  thrust::device_vector<int> d_num_items_vec(1, actual_num_items);
  const int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

  // Indirect sort
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  REQUIRE(
    cudaSuccess
    == (descending
          ? indirect_sort_keys_descending(
              d_temp_storage,
              temp_storage_bytes,
              thrust::raw_pointer_cast(d_keys_in.data()),
              thrust::raw_pointer_cast(d_keys_out_indirect.data()),
              d_num_items,
              max_num_items)
          : indirect_sort_keys(
              d_temp_storage,
              temp_storage_bytes,
              thrust::raw_pointer_cast(d_keys_in.data()),
              thrust::raw_pointer_cast(d_keys_out_indirect.data()),
              d_num_items,
              max_num_items)));
  REQUIRE(temp_storage_bytes > 0);

  thrust::device_vector<std::uint8_t> d_temp(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(d_temp.data());

  REQUIRE(
    cudaSuccess
    == (descending
          ? indirect_sort_keys_descending(
              d_temp_storage,
              temp_storage_bytes,
              thrust::raw_pointer_cast(d_keys_in.data()),
              thrust::raw_pointer_cast(d_keys_out_indirect.data()),
              d_num_items,
              max_num_items)
          : indirect_sort_keys(
              d_temp_storage,
              temp_storage_bytes,
              thrust::raw_pointer_cast(d_keys_in.data()),
              thrust::raw_pointer_cast(d_keys_out_indirect.data()),
              d_num_items,
              max_num_items)));
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  // Reference
  auto ref_keys = sort_reference(h_keys, actual_num_items, descending);

  thrust::host_vector<KeyT> h_keys_out_indirect(d_keys_out_indirect);
  for (int i = 0; i < actual_num_items; ++i)
  {
    REQUIRE(h_keys_out_indirect[i] == ref_keys[i]);
  }

  // Standard CUB sort comparison
  if (compare_with_standard && actual_num_items > 0)
  {
    size_t std_temp_bytes = 0;
    if (descending)
    {
      REQUIRE(
        cudaSuccess
        == cub::DeviceRadixSort::SortKeysDescending(
          nullptr,
          std_temp_bytes,
          thrust::raw_pointer_cast(d_keys_in.data()),
          thrust::raw_pointer_cast(d_keys_out_standard.data()),
          actual_num_items));
    }
    else
    {
      REQUIRE(
        cudaSuccess
        == cub::DeviceRadixSort::SortKeys(
          nullptr,
          std_temp_bytes,
          thrust::raw_pointer_cast(d_keys_in.data()),
          thrust::raw_pointer_cast(d_keys_out_standard.data()),
          actual_num_items));
    }

    thrust::device_vector<std::uint8_t> d_std_temp(std_temp_bytes);
    if (descending)
    {
      REQUIRE(
        cudaSuccess
        == cub::DeviceRadixSort::SortKeysDescending(
          thrust::raw_pointer_cast(d_std_temp.data()),
          std_temp_bytes,
          thrust::raw_pointer_cast(d_keys_in.data()),
          thrust::raw_pointer_cast(d_keys_out_standard.data()),
          actual_num_items));
    }
    else
    {
      REQUIRE(
        cudaSuccess
        == cub::DeviceRadixSort::SortKeys(
          thrust::raw_pointer_cast(d_std_temp.data()),
          std_temp_bytes,
          thrust::raw_pointer_cast(d_keys_in.data()),
          thrust::raw_pointer_cast(d_keys_out_standard.data()),
          actual_num_items));
    }
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    thrust::host_vector<KeyT> h_keys_out_standard(d_keys_out_standard);
    for (int i = 0; i < actual_num_items; ++i)
    {
      REQUIRE(h_keys_out_indirect[i] == h_keys_out_standard[i]);
    }
  }
}

template <typename KeyT, typename ValueT>
void verify_indirect_sort_pairs(
  const std::vector<KeyT>& h_keys,
  const std::vector<ValueT>& h_values,
  int max_num_items,
  int actual_num_items,
  bool descending,
  bool compare_with_standard = true)
{
  thrust::device_vector<KeyT> d_keys_in(h_keys.begin(), h_keys.end());
  thrust::device_vector<KeyT> d_keys_out_indirect(max_num_items);
  thrust::device_vector<ValueT> d_values_in(h_values.begin(), h_values.end());
  thrust::device_vector<ValueT> d_values_out_indirect(max_num_items);

  thrust::device_vector<int> d_num_items_vec(1, actual_num_items);
  const int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

  // Indirect sort
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  if (descending)
  {
    REQUIRE(
      cudaSuccess
      == indirect_sort_pairs_descending(
        d_temp_storage,
        temp_storage_bytes,
        thrust::raw_pointer_cast(d_keys_in.data()),
        thrust::raw_pointer_cast(d_keys_out_indirect.data()),
        thrust::raw_pointer_cast(d_values_in.data()),
        thrust::raw_pointer_cast(d_values_out_indirect.data()),
        d_num_items,
        max_num_items));
  }
  else
  {
    REQUIRE(
      cudaSuccess
      == indirect_sort_pairs(
        d_temp_storage,
        temp_storage_bytes,
        thrust::raw_pointer_cast(d_keys_in.data()),
        thrust::raw_pointer_cast(d_keys_out_indirect.data()),
        thrust::raw_pointer_cast(d_values_in.data()),
        thrust::raw_pointer_cast(d_values_out_indirect.data()),
        d_num_items,
        max_num_items));
  }
  REQUIRE(temp_storage_bytes > 0);

  thrust::device_vector<std::uint8_t> d_temp(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(d_temp.data());

  if (descending)
  {
    REQUIRE(
      cudaSuccess
      == indirect_sort_pairs_descending(
        d_temp_storage,
        temp_storage_bytes,
        thrust::raw_pointer_cast(d_keys_in.data()),
        thrust::raw_pointer_cast(d_keys_out_indirect.data()),
        thrust::raw_pointer_cast(d_values_in.data()),
        thrust::raw_pointer_cast(d_values_out_indirect.data()),
        d_num_items,
        max_num_items));
  }
  else
  {
    REQUIRE(
      cudaSuccess
      == indirect_sort_pairs(
        d_temp_storage,
        temp_storage_bytes,
        thrust::raw_pointer_cast(d_keys_in.data()),
        thrust::raw_pointer_cast(d_keys_out_indirect.data()),
        thrust::raw_pointer_cast(d_values_in.data()),
        thrust::raw_pointer_cast(d_values_out_indirect.data()),
        d_num_items,
        max_num_items));
  }
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  // Reference
  std::vector<KeyT> ref_keys;
  std::vector<ValueT> ref_values;
  sort_pairs_reference(h_keys, h_values, ref_keys, ref_values, actual_num_items, descending);

  thrust::host_vector<KeyT> h_keys_out_indirect(d_keys_out_indirect);
  thrust::host_vector<ValueT> h_values_out_indirect(d_values_out_indirect);
  for (int i = 0; i < actual_num_items; ++i)
  {
    REQUIRE(h_keys_out_indirect[i] == ref_keys[i]);
    REQUIRE(h_values_out_indirect[i] == ref_values[i]);
  }

  // Standard CUB sort comparison
  if (compare_with_standard && actual_num_items > 0)
  {
    thrust::device_vector<KeyT> d_keys_out_standard(max_num_items);
    thrust::device_vector<ValueT> d_values_out_standard(max_num_items);

    size_t std_temp_bytes = 0;
    if (descending)
    {
      REQUIRE(
        cudaSuccess
        == cub::DeviceRadixSort::SortPairsDescending(
          nullptr,
          std_temp_bytes,
          thrust::raw_pointer_cast(d_keys_in.data()),
          thrust::raw_pointer_cast(d_keys_out_standard.data()),
          thrust::raw_pointer_cast(d_values_in.data()),
          thrust::raw_pointer_cast(d_values_out_standard.data()),
          actual_num_items));
    }
    else
    {
      REQUIRE(
        cudaSuccess
        == cub::DeviceRadixSort::SortPairs(
          nullptr,
          std_temp_bytes,
          thrust::raw_pointer_cast(d_keys_in.data()),
          thrust::raw_pointer_cast(d_keys_out_standard.data()),
          thrust::raw_pointer_cast(d_values_in.data()),
          thrust::raw_pointer_cast(d_values_out_standard.data()),
          actual_num_items));
    }

    thrust::device_vector<std::uint8_t> d_std_temp(std_temp_bytes);
    if (descending)
    {
      REQUIRE(
        cudaSuccess
        == cub::DeviceRadixSort::SortPairsDescending(
          thrust::raw_pointer_cast(d_std_temp.data()),
          std_temp_bytes,
          thrust::raw_pointer_cast(d_keys_in.data()),
          thrust::raw_pointer_cast(d_keys_out_standard.data()),
          thrust::raw_pointer_cast(d_values_in.data()),
          thrust::raw_pointer_cast(d_values_out_standard.data()),
          actual_num_items));
    }
    else
    {
      REQUIRE(
        cudaSuccess
        == cub::DeviceRadixSort::SortPairs(
          thrust::raw_pointer_cast(d_std_temp.data()),
          std_temp_bytes,
          thrust::raw_pointer_cast(d_keys_in.data()),
          thrust::raw_pointer_cast(d_keys_out_standard.data()),
          thrust::raw_pointer_cast(d_values_in.data()),
          thrust::raw_pointer_cast(d_values_out_standard.data()),
          actual_num_items));
    }
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    thrust::host_vector<KeyT> h_keys_out_standard(d_keys_out_standard);
    thrust::host_vector<ValueT> h_values_out_standard(d_values_out_standard);
    for (int i = 0; i < actual_num_items; ++i)
    {
      REQUIRE(h_keys_out_indirect[i] == h_keys_out_standard[i]);
      REQUIRE(h_values_out_indirect[i] == h_values_out_standard[i]);
    }
  }
}

// ============================================================================
// SortPairs ascending
// ============================================================================

TEST_CASE("DeviceRadixSort::SortPairs indirect basic", "[radix_sort][indirect][device]")
{
  constexpr int max_num_items = 2000;
  const int actual_num_items  = GENERATE(0, 1, 42, 500, 2000);

  std::vector<int> h_keys(max_num_items);
  std::vector<int> h_values(max_num_items);
  for (int i = 0; i < max_num_items; ++i)
  {
    h_keys[i]   = max_num_items - i;
    h_values[i] = i;
  }

  verify_indirect_sort_pairs(h_keys, h_values, max_num_items, actual_num_items, false);
}

TEST_CASE("DeviceRadixSort::SortPairs indirect random data", "[radix_sort][indirect][device]")
{
  constexpr int max_num_items = 50000;
  const int actual_num_items  = GENERATE(100, 1000, 10000, 50000);

  std::mt19937 rng(42);
  std::uniform_int_distribution<int> key_dist(0, 10000);

  std::vector<int> h_keys(max_num_items);
  std::vector<int> h_values(max_num_items);
  for (int i = 0; i < max_num_items; ++i)
  {
    h_keys[i]   = key_dist(rng);
    h_values[i] = i;
  }

  verify_indirect_sort_pairs(h_keys, h_values, max_num_items, actual_num_items, false);
}

TEST_CASE("DeviceRadixSort::SortPairs indirect large", "[radix_sort][indirect][device]")
{
  constexpr int max_num_items = 1 << 20;
  const int actual_num_items  = GENERATE(1 << 18, 1 << 19, 1 << 20);

  std::mt19937 rng(7);
  std::uniform_int_distribution<int> dist(0, std::numeric_limits<int>::max());

  std::vector<int> h_keys(max_num_items);
  std::vector<int> h_values(max_num_items);
  for (int i = 0; i < max_num_items; ++i)
  {
    h_keys[i]   = dist(rng);
    h_values[i] = i;
  }

  verify_indirect_sort_pairs(h_keys, h_values, max_num_items, actual_num_items, false);
}

// ============================================================================
// SortPairsDescending
// ============================================================================

TEST_CASE("DeviceRadixSort::SortPairsDescending indirect basic", "[radix_sort][indirect][device]")
{
  constexpr int max_num_items = 1000;
  const int actual_num_items  = GENERATE(0, 1, 100, 1000);

  std::vector<int> h_keys(max_num_items);
  std::vector<int> h_values(max_num_items);
  for (int i = 0; i < max_num_items; ++i)
  {
    h_keys[i]   = i;
    h_values[i] = i * 10;
  }

  verify_indirect_sort_pairs(h_keys, h_values, max_num_items, actual_num_items, true);
}

TEST_CASE("DeviceRadixSort::SortPairsDescending indirect random data", "[radix_sort][indirect][device]")
{
  constexpr int max_num_items = 50000;
  const int actual_num_items  = GENERATE(1, 1000, 50000);

  std::mt19937 rng(99);
  std::uniform_int_distribution<int> key_dist(0, 100000);

  std::vector<int> h_keys(max_num_items);
  std::vector<int> h_values(max_num_items);
  for (int i = 0; i < max_num_items; ++i)
  {
    h_keys[i]   = key_dist(rng);
    h_values[i] = i;
  }

  verify_indirect_sort_pairs(h_keys, h_values, max_num_items, actual_num_items, true);
}

// ============================================================================
// SortKeys ascending
// ============================================================================

TEST_CASE("DeviceRadixSort::SortKeys indirect basic", "[radix_sort][indirect][device]")
{
  constexpr int max_num_items = 1000;
  const int actual_num_items  = GENERATE(0, 1, 100, 1000);

  std::vector<int> h_keys(max_num_items);
  for (int i = 0; i < max_num_items; ++i)
  {
    h_keys[i] = max_num_items - i;
  }

  verify_indirect_sort_keys(h_keys, max_num_items, actual_num_items, false);
}

TEST_CASE("DeviceRadixSort::SortKeys indirect random data", "[radix_sort][indirect][device]")
{
  constexpr int max_num_items = 50000;
  const int actual_num_items  = GENERATE(100, 10000, 50000);

  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, std::numeric_limits<int>::max());

  std::vector<int> h_keys(max_num_items);
  for (auto& k : h_keys)
  {
    k = dist(rng);
  }

  verify_indirect_sort_keys(h_keys, max_num_items, actual_num_items, false);
}

TEST_CASE("DeviceRadixSort::SortKeys indirect large", "[radix_sort][indirect][device]")
{
  constexpr int max_num_items = 1 << 20;
  const int actual_num_items  = GENERATE(1 << 18, 1 << 20);

  std::mt19937 rng(7);
  std::uniform_int_distribution<int> dist(0, std::numeric_limits<int>::max());

  std::vector<int> h_keys(max_num_items);
  for (auto& k : h_keys)
  {
    k = dist(rng);
  }

  verify_indirect_sort_keys(h_keys, max_num_items, actual_num_items, false);
}

// ============================================================================
// SortKeysDescending
// ============================================================================

TEST_CASE("DeviceRadixSort::SortKeysDescending indirect basic", "[radix_sort][indirect][device]")
{
  constexpr int max_num_items = 1000;
  const int actual_num_items  = GENERATE(0, 1, 100, 1000);

  std::vector<int> h_keys(max_num_items);
  for (int i = 0; i < max_num_items; ++i)
  {
    h_keys[i] = i;
  }

  verify_indirect_sort_keys(h_keys, max_num_items, actual_num_items, true);
}

TEST_CASE("DeviceRadixSort::SortKeysDescending indirect random", "[radix_sort][indirect][device]")
{
  constexpr int max_num_items = 50000;
  const int actual_num_items  = GENERATE(1, 10000, 50000);

  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, std::numeric_limits<int>::max());

  std::vector<int> h_keys(max_num_items);
  for (auto& k : h_keys)
  {
    k = dist(rng);
  }

  verify_indirect_sort_keys(h_keys, max_num_items, actual_num_items, true);
}

// ============================================================================
// Float key types
// ============================================================================

TEST_CASE("DeviceRadixSort::SortPairs indirect with float keys", "[radix_sort][indirect][device]")
{
  constexpr int max_num_items = 10000;
  const int actual_num_items  = GENERATE(0, 1, 200, 5000, 10000);

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);

  std::vector<float> h_keys(max_num_items);
  std::vector<int> h_values(max_num_items);
  for (int i = 0; i < max_num_items; ++i)
  {
    h_keys[i]   = dist(rng);
    h_values[i] = i;
  }

  verify_indirect_sort_pairs(h_keys, h_values, max_num_items, actual_num_items, false);
}

TEST_CASE("DeviceRadixSort::SortKeys indirect with float", "[radix_sort][indirect][device]")
{
  constexpr int max_num_items = 10000;
  const int actual_num_items  = GENERATE(1, 5000, 10000);

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);

  std::vector<float> h_keys(max_num_items);
  for (auto& k : h_keys)
  {
    k = dist(rng);
  }

  verify_indirect_sort_keys(h_keys, max_num_items, actual_num_items, false);
}

TEST_CASE("DeviceRadixSort::SortKeys indirect with double", "[radix_sort][indirect][device]")
{
  constexpr int max_num_items = 10000;
  const int actual_num_items  = GENERATE(1, 5000, 10000);

  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(-1e6, 1e6);

  std::vector<double> h_keys(max_num_items);
  for (auto& k : h_keys)
  {
    k = dist(rng);
  }

  verify_indirect_sort_keys(h_keys, max_num_items, actual_num_items, false);
}

// ============================================================================
// Unsigned key types
// ============================================================================

TEST_CASE("DeviceRadixSort::SortKeys indirect uint8_t", "[radix_sort][indirect][device]")
{
  constexpr int max_num_items = 10000;
  const int actual_num_items  = GENERATE(1, 1000, 10000);

  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, 255);

  std::vector<std::uint8_t> h_keys(max_num_items);
  for (auto& k : h_keys)
  {
    k = static_cast<std::uint8_t>(dist(rng));
  }

  verify_indirect_sort_keys(h_keys, max_num_items, actual_num_items, false);
}

TEST_CASE("DeviceRadixSort::SortKeys indirect uint16_t", "[radix_sort][indirect][device]")
{
  constexpr int max_num_items = 10000;
  const int actual_num_items  = GENERATE(1, 5000, 10000);

  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, 65535);

  std::vector<std::uint16_t> h_keys(max_num_items);
  for (auto& k : h_keys)
  {
    k = static_cast<std::uint16_t>(dist(rng));
  }

  verify_indirect_sort_keys(h_keys, max_num_items, actual_num_items, false);
}

TEST_CASE("DeviceRadixSort::SortKeys indirect uint32_t", "[radix_sort][indirect][device]")
{
  constexpr int max_num_items = 10000;
  const int actual_num_items  = GENERATE(1, 5000, 10000);

  std::mt19937 rng(42);
  std::uniform_int_distribution<std::uint32_t> dist(0, std::numeric_limits<std::uint32_t>::max());

  std::vector<std::uint32_t> h_keys(max_num_items);
  for (auto& k : h_keys)
  {
    k = dist(rng);
  }

  verify_indirect_sort_keys(h_keys, max_num_items, actual_num_items, false);
}

TEST_CASE("DeviceRadixSort::SortKeys indirect uint64_t", "[radix_sort][indirect][device]")
{
  constexpr int max_num_items = 10000;
  const int actual_num_items  = GENERATE(1, 5000, 10000);

  std::mt19937 rng(42);
  std::uniform_int_distribution<std::uint64_t> dist(0, std::numeric_limits<std::uint64_t>::max());

  std::vector<std::uint64_t> h_keys(max_num_items);
  for (auto& k : h_keys)
  {
    k = dist(rng);
  }

  verify_indirect_sort_keys(h_keys, max_num_items, actual_num_items, false);
}

// ============================================================================
// Duplicate keys (tests stability)
// ============================================================================

TEST_CASE("DeviceRadixSort::SortPairs indirect all duplicate keys", "[radix_sort][indirect][device]")
{
  constexpr int max_num_items = 5000;
  const int actual_num_items  = GENERATE(1, 100, 5000);

  std::vector<int> h_keys(max_num_items, 42); // all same key
  std::vector<int> h_values(max_num_items);
  std::iota(h_values.begin(), h_values.end(), 0);

  verify_indirect_sort_pairs(h_keys, h_values, max_num_items, actual_num_items, false);
}

TEST_CASE("DeviceRadixSort::SortPairs indirect few unique keys", "[radix_sort][indirect][device]")
{
  constexpr int max_num_items = 10000;
  const int actual_num_items  = GENERATE(100, 5000, 10000);

  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, 3); // only 4 unique keys

  std::vector<int> h_keys(max_num_items);
  std::vector<int> h_values(max_num_items);
  for (int i = 0; i < max_num_items; ++i)
  {
    h_keys[i]   = dist(rng);
    h_values[i] = i;
  }

  verify_indirect_sort_pairs(h_keys, h_values, max_num_items, actual_num_items, false);
}

// ============================================================================
// Already sorted / reverse sorted
// ============================================================================

TEST_CASE("DeviceRadixSort::SortKeys indirect already sorted ascending", "[radix_sort][indirect][device]")
{
  constexpr int max_num_items = 10000;
  const int actual_num_items  = GENERATE(1, 5000, 10000);

  std::vector<int> h_keys(max_num_items);
  std::iota(h_keys.begin(), h_keys.end(), 0);

  verify_indirect_sort_keys(h_keys, max_num_items, actual_num_items, false);
}

TEST_CASE("DeviceRadixSort::SortKeys indirect reverse sorted", "[radix_sort][indirect][device]")
{
  constexpr int max_num_items = 10000;
  const int actual_num_items  = GENERATE(1, 5000, 10000);

  std::vector<int> h_keys(max_num_items);
  for (int i = 0; i < max_num_items; ++i)
  {
    h_keys[i] = max_num_items - i;
  }

  verify_indirect_sort_keys(h_keys, max_num_items, actual_num_items, false);
}

// ============================================================================
// Power-of-two and non-power-of-two sizes
// ============================================================================

TEST_CASE("DeviceRadixSort::SortKeys indirect power-of-two sizes", "[radix_sort][indirect][device]")
{
  constexpr int max_num_items = 1 << 16;
  const int actual_num_items  = GENERATE(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1 << 16);

  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, 10000);

  std::vector<int> h_keys(max_num_items);
  for (auto& k : h_keys)
  {
    k = dist(rng);
  }

  verify_indirect_sort_keys(h_keys, max_num_items, actual_num_items, false);
}

TEST_CASE("DeviceRadixSort::SortKeys indirect non-power-of-two sizes", "[radix_sort][indirect][device]")
{
  constexpr int max_num_items = 10000;
  const int actual_num_items  = GENERATE(3, 7, 13, 127, 129, 255, 257, 511, 513, 1023, 1025, 4095, 4097, 9999);

  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, 10000);

  std::vector<int> h_keys(max_num_items);
  for (auto& k : h_keys)
  {
    k = dist(rng);
  }

  verify_indirect_sort_keys(h_keys, max_num_items, actual_num_items, false);
}

// ============================================================================
// actual_num_items much smaller than max
// ============================================================================

TEST_CASE("DeviceRadixSort::SortKeys indirect actual much smaller than max", "[radix_sort][indirect][device]")
{
  constexpr int max_num_items = 1 << 18;
  const int actual_num_items  = GENERATE(0, 1, 10, 100);

  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, 10000);

  std::vector<int> h_keys(max_num_items);
  for (auto& k : h_keys)
  {
    k = dist(rng);
  }

  verify_indirect_sort_keys(h_keys, max_num_items, actual_num_items, false);
}

// ============================================================================
// Partial bit range
// ============================================================================

TEST_CASE("DeviceRadixSort::SortKeys indirect partial bit range", "[radix_sort][indirect][device]")
{
  constexpr int max_num_items = 10000;
  const int actual_num_items  = GENERATE(1, 5000, 10000);

  // Keys with values 0-255, but we only sort on lower 4 bits
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, 255);

  std::vector<int> h_keys(max_num_items);
  std::vector<int> h_values(max_num_items);
  for (int i = 0; i < max_num_items; ++i)
  {
    h_keys[i]   = dist(rng);
    h_values[i] = i;
  }

  thrust::device_vector<int> d_keys_in(h_keys.begin(), h_keys.end());
  thrust::device_vector<int> d_keys_out(max_num_items);
  thrust::device_vector<int> d_values_in(h_values.begin(), h_values.end());
  thrust::device_vector<int> d_values_out(max_num_items);

  thrust::device_vector<int> d_num_items_vec(1, actual_num_items);
  const int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

  const int begin_bit = 0;
  const int end_bit   = 4;

  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortPairs(
      d_temp_storage,
      temp_storage_bytes,
      thrust::raw_pointer_cast(d_keys_in.data()),
      thrust::raw_pointer_cast(d_keys_out.data()),
      thrust::raw_pointer_cast(d_values_in.data()),
      thrust::raw_pointer_cast(d_values_out.data()),
      d_num_items,
      max_num_items,
      begin_bit,
      end_bit));

  thrust::device_vector<std::uint8_t> d_temp(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(d_temp.data());

  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortPairs(
      d_temp_storage,
      temp_storage_bytes,
      thrust::raw_pointer_cast(d_keys_in.data()),
      thrust::raw_pointer_cast(d_keys_out.data()),
      thrust::raw_pointer_cast(d_values_in.data()),
      thrust::raw_pointer_cast(d_values_out.data()),
      d_num_items,
      max_num_items,
      begin_bit,
      end_bit));
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  // Reference: sort by lower 4 bits using stable_sort
  std::vector<std::pair<int, int>> pairs(actual_num_items);
  for (int i = 0; i < actual_num_items; ++i)
  {
    pairs[i] = {h_keys[i], h_values[i]};
  }
  std::stable_sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) {
    return (a.first & 0xF) < (b.first & 0xF);
  });

  thrust::host_vector<int> h_keys_out(d_keys_out);
  thrust::host_vector<int> h_values_out(d_values_out);
  for (int i = 0; i < actual_num_items; ++i)
  {
    REQUIRE((h_keys_out[i] & 0xF) == (pairs[i].first & 0xF));
    REQUIRE(h_values_out[i] == pairs[i].second);
  }
}

// ============================================================================
// Varying num_items between calls (reusing temp storage)
// ============================================================================

TEST_CASE("DeviceRadixSort indirect SortPairs varying num_items between calls", "[radix_sort][indirect][device]")
{
  constexpr int max_num_items = 20000;

  std::mt19937 rng(42);
  std::uniform_int_distribution<int> key_dist(0, 100000);

  std::vector<int> h_keys(max_num_items);
  std::vector<int> h_values(max_num_items);
  for (int i = 0; i < max_num_items; ++i)
  {
    h_keys[i]   = key_dist(rng);
    h_values[i] = i;
  }
  thrust::device_vector<int> d_keys_in(h_keys.begin(), h_keys.end());
  thrust::device_vector<int> d_keys_out(max_num_items, -1);
  thrust::device_vector<int> d_values_in(h_values.begin(), h_values.end());
  thrust::device_vector<int> d_values_out(max_num_items, -1);

  thrust::device_vector<int> d_num_items_vec(1, 0);
  int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  REQUIRE(
    cudaSuccess
    == indirect_sort_pairs(
      d_temp_storage,
      temp_storage_bytes,
      thrust::raw_pointer_cast(d_keys_in.data()),
      thrust::raw_pointer_cast(d_keys_out.data()),
      thrust::raw_pointer_cast(d_values_in.data()),
      thrust::raw_pointer_cast(d_values_out.data()),
      static_cast<const int*>(d_num_items),
      max_num_items));

  thrust::device_vector<std::uint8_t> d_temp(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(d_temp.data());

  for (int n : {10, 500, 1, 20000, 0, 42, 19999, 1024, 1023})
  {
    d_num_items_vec[0] = n;
    thrust::fill(d_keys_out.begin(), d_keys_out.end(), -1);
    thrust::fill(d_values_out.begin(), d_values_out.end(), -1);

    REQUIRE(
      cudaSuccess
      == indirect_sort_pairs(
        d_temp_storage,
        temp_storage_bytes,
        thrust::raw_pointer_cast(d_keys_in.data()),
        thrust::raw_pointer_cast(d_keys_out.data()),
        thrust::raw_pointer_cast(d_values_in.data()),
        thrust::raw_pointer_cast(d_values_out.data()),
        static_cast<const int*>(d_num_items),
        max_num_items));
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    std::vector<int> ref_keys, ref_values;
    sort_pairs_reference(h_keys, h_values, ref_keys, ref_values, n, false);

    thrust::host_vector<int> h_keys_out(d_keys_out);
    thrust::host_vector<int> h_values_out(d_values_out);
    for (int i = 0; i < n; ++i)
    {
      REQUIRE(h_keys_out[i] == ref_keys[i]);
      REQUIRE(h_values_out[i] == ref_values[i]);
    }
  }
}

TEST_CASE("DeviceRadixSort indirect SortKeys varying num_items between calls", "[radix_sort][indirect][device]")
{
  constexpr int max_num_items = 20000;

  std::mt19937 rng(42);
  std::uniform_int_distribution<int> key_dist(0, 100000);

  std::vector<int> h_keys(max_num_items);
  for (auto& k : h_keys)
  {
    k = key_dist(rng);
  }
  thrust::device_vector<int> d_keys_in(h_keys.begin(), h_keys.end());
  thrust::device_vector<int> d_keys_out(max_num_items, -1);

  thrust::device_vector<int> d_num_items_vec(1, 0);
  int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  REQUIRE(
    cudaSuccess
    == indirect_sort_keys(
      d_temp_storage,
      temp_storage_bytes,
      thrust::raw_pointer_cast(d_keys_in.data()),
      thrust::raw_pointer_cast(d_keys_out.data()),
      static_cast<const int*>(d_num_items),
      max_num_items));

  thrust::device_vector<std::uint8_t> d_temp(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(d_temp.data());

  for (int n : {10, 500, 1, 20000, 0, 42, 19999})
  {
    d_num_items_vec[0] = n;
    thrust::fill(d_keys_out.begin(), d_keys_out.end(), -1);

    REQUIRE(
      cudaSuccess
      == indirect_sort_keys(
        d_temp_storage,
        temp_storage_bytes,
        thrust::raw_pointer_cast(d_keys_in.data()),
        thrust::raw_pointer_cast(d_keys_out.data()),
        static_cast<const int*>(d_num_items),
        max_num_items));
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    auto ref_keys = sort_reference(h_keys, n, false);

    thrust::host_vector<int> h_keys_out(d_keys_out);
    for (int i = 0; i < n; ++i)
    {
      REQUIRE(h_keys_out[i] == ref_keys[i]);
    }
  }
}

// ============================================================================
// CUDA graph capture tests
// ============================================================================

TEST_CASE("DeviceRadixSort::SortPairs indirect CUDA graph capture", "[radix_sort][indirect][device][graph]")
{
  constexpr int max_num_items = 10000;

  std::mt19937 rng(42);
  std::uniform_int_distribution<int> key_dist(0, 100000);

  std::vector<int> h_keys(max_num_items);
  std::vector<int> h_values(max_num_items);
  for (int i = 0; i < max_num_items; ++i)
  {
    h_keys[i]   = key_dist(rng);
    h_values[i] = i;
  }
  thrust::device_vector<int> d_keys_in(h_keys.begin(), h_keys.end());
  thrust::device_vector<int> d_keys_out(max_num_items, -1);
  thrust::device_vector<int> d_values_in(h_values.begin(), h_values.end());
  thrust::device_vector<int> d_values_out(max_num_items, -1);

  thrust::device_vector<int> d_num_items_vec(1, 0);
  int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

  // Allocate temp storage
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  REQUIRE(
    cudaSuccess
    == indirect_sort_pairs(
      d_temp_storage,
      temp_storage_bytes,
      thrust::raw_pointer_cast(d_keys_in.data()),
      thrust::raw_pointer_cast(d_keys_out.data()),
      thrust::raw_pointer_cast(d_values_in.data()),
      thrust::raw_pointer_cast(d_values_out.data()),
      static_cast<const int*>(d_num_items),
      max_num_items));

  thrust::device_vector<std::uint8_t> d_temp(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(d_temp.data());

  // Capture graph
  cudaStream_t stream{};
  REQUIRE(cudaSuccess == cudaStreamCreate(&stream));

  cudaGraph_t graph{};
  REQUIRE(cudaSuccess == cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortPairs(
      d_temp_storage,
      temp_storage_bytes,
      thrust::raw_pointer_cast(d_keys_in.data()),
      thrust::raw_pointer_cast(d_keys_out.data()),
      thrust::raw_pointer_cast(d_values_in.data()),
      thrust::raw_pointer_cast(d_values_out.data()),
      static_cast<const int*>(d_num_items),
      max_num_items,
      0, // begin_bit
      static_cast<int>(sizeof(int) * 8), // end_bit
      stream));

  REQUIRE(cudaSuccess == cudaStreamEndCapture(stream, &graph));

  cudaGraphExec_t exec{};
  REQUIRE(cudaSuccess == cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  // Run the graph with different num_items values
  for (int n : {0, 1, 42, 1000, 5000, 10000})
  {
    d_num_items_vec[0] = n;
    thrust::fill(d_keys_out.begin(), d_keys_out.end(), -1);
    thrust::fill(d_values_out.begin(), d_values_out.end(), -1);
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    REQUIRE(cudaSuccess == cudaGraphLaunch(exec, stream));
    REQUIRE(cudaSuccess == cudaStreamSynchronize(stream));

    std::vector<int> ref_keys, ref_values;
    sort_pairs_reference(h_keys, h_values, ref_keys, ref_values, n, false);

    thrust::host_vector<int> h_keys_out(d_keys_out);
    thrust::host_vector<int> h_values_out(d_values_out);
    for (int i = 0; i < n; ++i)
    {
      REQUIRE(h_keys_out[i] == ref_keys[i]);
      REQUIRE(h_values_out[i] == ref_values[i]);
    }
  }

  REQUIRE(cudaSuccess == cudaGraphExecDestroy(exec));
  REQUIRE(cudaSuccess == cudaGraphDestroy(graph));
  REQUIRE(cudaSuccess == cudaStreamDestroy(stream));
}

TEST_CASE("DeviceRadixSort::SortKeys indirect CUDA graph capture", "[radix_sort][indirect][device][graph]")
{
  constexpr int max_num_items = 10000;

  std::mt19937 rng(42);
  std::uniform_int_distribution<int> key_dist(0, 100000);

  std::vector<int> h_keys(max_num_items);
  for (auto& k : h_keys)
  {
    k = key_dist(rng);
  }
  thrust::device_vector<int> d_keys_in(h_keys.begin(), h_keys.end());
  thrust::device_vector<int> d_keys_out(max_num_items, -1);

  thrust::device_vector<int> d_num_items_vec(1, 0);
  int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  REQUIRE(
    cudaSuccess
    == indirect_sort_keys(
      d_temp_storage,
      temp_storage_bytes,
      thrust::raw_pointer_cast(d_keys_in.data()),
      thrust::raw_pointer_cast(d_keys_out.data()),
      static_cast<const int*>(d_num_items),
      max_num_items));

  thrust::device_vector<std::uint8_t> d_temp(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(d_temp.data());

  cudaStream_t stream{};
  REQUIRE(cudaSuccess == cudaStreamCreate(&stream));

  cudaGraph_t graph{};
  REQUIRE(cudaSuccess == cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortKeys(
      d_temp_storage,
      temp_storage_bytes,
      thrust::raw_pointer_cast(d_keys_in.data()),
      thrust::raw_pointer_cast(d_keys_out.data()),
      static_cast<const int*>(d_num_items),
      max_num_items,
      0,
      static_cast<int>(sizeof(int) * 8),
      stream));

  REQUIRE(cudaSuccess == cudaStreamEndCapture(stream, &graph));

  cudaGraphExec_t exec{};
  REQUIRE(cudaSuccess == cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  for (int n : {0, 1, 42, 1000, 5000, 10000})
  {
    d_num_items_vec[0] = n;
    thrust::fill(d_keys_out.begin(), d_keys_out.end(), -1);
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    REQUIRE(cudaSuccess == cudaGraphLaunch(exec, stream));
    REQUIRE(cudaSuccess == cudaStreamSynchronize(stream));

    auto ref_keys = sort_reference(h_keys, n, false);

    thrust::host_vector<int> h_keys_out(d_keys_out);
    for (int i = 0; i < n; ++i)
    {
      REQUIRE(h_keys_out[i] == ref_keys[i]);
    }
  }

  REQUIRE(cudaSuccess == cudaGraphExecDestroy(exec));
  REQUIRE(cudaSuccess == cudaGraphDestroy(graph));
  REQUIRE(cudaSuccess == cudaStreamDestroy(stream));
}

// ============================================================================
// Graph capture stress test
// ============================================================================

TEST_CASE("DeviceRadixSort indirect graph capture stress", "[radix_sort][indirect][device][graph][stress]")
{
  constexpr int max_num_items = 50000;

  std::mt19937 data_rng(42);
  std::uniform_int_distribution<int> key_dist(0, 1000000);

  std::vector<int> h_keys(max_num_items);
  std::vector<int> h_values(max_num_items);
  for (int i = 0; i < max_num_items; ++i)
  {
    h_keys[i]   = key_dist(data_rng);
    h_values[i] = i;
  }
  thrust::device_vector<int> d_keys_in(h_keys.begin(), h_keys.end());
  thrust::device_vector<int> d_keys_out(max_num_items, -1);
  thrust::device_vector<int> d_values_in(h_values.begin(), h_values.end());
  thrust::device_vector<int> d_values_out(max_num_items, -1);

  thrust::device_vector<int> d_num_items_vec(1, 0);
  int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  REQUIRE(
    cudaSuccess
    == indirect_sort_pairs(
      d_temp_storage,
      temp_storage_bytes,
      thrust::raw_pointer_cast(d_keys_in.data()),
      thrust::raw_pointer_cast(d_keys_out.data()),
      thrust::raw_pointer_cast(d_values_in.data()),
      thrust::raw_pointer_cast(d_values_out.data()),
      static_cast<const int*>(d_num_items),
      max_num_items));

  thrust::device_vector<std::uint8_t> d_temp(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(d_temp.data());

  cudaStream_t stream{};
  REQUIRE(cudaSuccess == cudaStreamCreate(&stream));

  cudaGraph_t graph{};
  REQUIRE(cudaSuccess == cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortPairs(
      d_temp_storage,
      temp_storage_bytes,
      thrust::raw_pointer_cast(d_keys_in.data()),
      thrust::raw_pointer_cast(d_keys_out.data()),
      thrust::raw_pointer_cast(d_values_in.data()),
      thrust::raw_pointer_cast(d_values_out.data()),
      static_cast<const int*>(d_num_items),
      max_num_items,
      0,
      static_cast<int>(sizeof(int) * 8),
      stream));

  REQUIRE(cudaSuccess == cudaStreamEndCapture(stream, &graph));

  cudaGraphExec_t exec{};
  REQUIRE(cudaSuccess == cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  // 50 iterations with pseudo-random sizes
  std::mt19937 size_rng(7);
  std::uniform_int_distribution<int> size_dist(0, max_num_items);

  for (int iter = 0; iter < 50; ++iter)
  {
    int n              = size_dist(size_rng);
    d_num_items_vec[0] = n;
    thrust::fill(d_keys_out.begin(), d_keys_out.end(), -1);
    thrust::fill(d_values_out.begin(), d_values_out.end(), -1);
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    REQUIRE(cudaSuccess == cudaGraphLaunch(exec, stream));
    REQUIRE(cudaSuccess == cudaStreamSynchronize(stream));

    std::vector<int> ref_keys, ref_values;
    sort_pairs_reference(h_keys, h_values, ref_keys, ref_values, n, false);

    thrust::host_vector<int> h_keys_out(d_keys_out);
    thrust::host_vector<int> h_values_out(d_values_out);
    for (int i = 0; i < n; ++i)
    {
      REQUIRE(h_keys_out[i] == ref_keys[i]);
      REQUIRE(h_values_out[i] == ref_values[i]);
    }
  }

  REQUIRE(cudaSuccess == cudaGraphExecDestroy(exec));
  REQUIRE(cudaSuccess == cudaGraphDestroy(graph));
  REQUIRE(cudaSuccess == cudaStreamDestroy(stream));
}

// ============================================================================
// Edge cases
// ============================================================================

TEST_CASE("DeviceRadixSort indirect zero max_num_items", "[radix_sort][indirect][device]")
{
  thrust::device_vector<int> d_keys_in(1, 0);
  thrust::device_vector<int> d_keys_out(1, 0);

  thrust::device_vector<int> d_num_items_vec(1, 0);
  const int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;

  // max_num_items = 0 should allocate minimal temp and return early
  REQUIRE(
    cudaSuccess
    == indirect_sort_keys(
      d_temp_storage,
      temp_storage_bytes,
      thrust::raw_pointer_cast(d_keys_in.data()),
      thrust::raw_pointer_cast(d_keys_out.data()),
      d_num_items,
      0));
}

TEST_CASE("DeviceRadixSort indirect single element", "[radix_sort][indirect][device]")
{
  constexpr int max_num_items = 100;
  const int actual_num_items  = 1;

  SECTION("SortKeys ascending")
  {
    std::vector<int> h_keys(max_num_items, 42);
    verify_indirect_sort_keys(h_keys, max_num_items, actual_num_items, false);
  }

  SECTION("SortKeys descending")
  {
    std::vector<int> h_keys(max_num_items, 42);
    verify_indirect_sort_keys(h_keys, max_num_items, actual_num_items, true);
  }

  SECTION("SortPairs ascending")
  {
    std::vector<int> h_keys(max_num_items, 42);
    std::vector<int> h_values(max_num_items, 99);
    verify_indirect_sort_pairs(h_keys, h_values, max_num_items, actual_num_items, false);
  }
}

// ============================================================================
// Negative key values (signed types)
// ============================================================================

TEST_CASE("DeviceRadixSort::SortKeys indirect negative values", "[radix_sort][indirect][device]")
{
  constexpr int max_num_items = 10000;
  const int actual_num_items  = GENERATE(1, 5000, 10000);

  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(std::numeric_limits<int>::min(), std::numeric_limits<int>::max());

  std::vector<int> h_keys(max_num_items);
  for (auto& k : h_keys)
  {
    k = dist(rng);
  }

  verify_indirect_sort_keys(h_keys, max_num_items, actual_num_items, false);
}

TEST_CASE("DeviceRadixSort::SortKeysDescending indirect negative values", "[radix_sort][indirect][device]")
{
  constexpr int max_num_items = 10000;
  const int actual_num_items  = GENERATE(1, 5000, 10000);

  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(std::numeric_limits<int>::min(), std::numeric_limits<int>::max());

  std::vector<int> h_keys(max_num_items);
  for (auto& k : h_keys)
  {
    k = dist(rng);
  }

  verify_indirect_sort_keys(h_keys, max_num_items, actual_num_items, true);
}

TEST_CASE("DeviceRadixSort::SortKeys indirect float with negatives", "[radix_sort][indirect][device]")
{
  constexpr int max_num_items = 10000;
  const int actual_num_items  = GENERATE(1, 5000, 10000);

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1e6f, 1e6f);

  std::vector<float> h_keys(max_num_items);
  for (auto& k : h_keys)
  {
    k = dist(rng);
  }

  verify_indirect_sort_keys(h_keys, max_num_items, actual_num_items, false);
  verify_indirect_sort_keys(h_keys, max_num_items, actual_num_items, true);
}
