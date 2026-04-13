// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_radix_sort.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <numeric>
#include <vector>

#include <c2h/catch2_test_helper.h>

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
    std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) {
      return a.first > b.first;
    });
  }
  else
  {
    std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) {
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

TEST_CASE("DeviceRadixSort::SortPairs indirect num_items", "[radix_sort][device]")
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
  thrust::device_vector<int> d_keys_in(h_keys.begin(), h_keys.end());
  thrust::device_vector<int> d_keys_out(max_num_items, -1);
  thrust::device_vector<int> d_values_in(h_values.begin(), h_values.end());
  thrust::device_vector<int> d_values_out(max_num_items, -1);

  thrust::device_vector<int> d_num_items_vec(1, actual_num_items);
  const int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

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
      0,
      static_cast<int>(sizeof(int) * 8)));
  REQUIRE(temp_storage_bytes > 0);

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
      0,
      static_cast<int>(sizeof(int) * 8)));

  std::vector<int> ref_keys, ref_values;
  sort_pairs_reference(h_keys, h_values, ref_keys, ref_values, actual_num_items, false);

  thrust::host_vector<int> h_keys_out(d_keys_out);
  thrust::host_vector<int> h_values_out(d_values_out);
  for (int i = 0; i < actual_num_items; ++i)
  {
    REQUIRE(h_keys_out[i] == ref_keys[i]);
    REQUIRE(h_values_out[i] == ref_values[i]);
  }
}

TEST_CASE("DeviceRadixSort::SortPairsDescending indirect num_items", "[radix_sort][device]")
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
  thrust::device_vector<int> d_keys_in(h_keys.begin(), h_keys.end());
  thrust::device_vector<int> d_keys_out(max_num_items, -1);
  thrust::device_vector<int> d_values_in(h_values.begin(), h_values.end());
  thrust::device_vector<int> d_values_out(max_num_items, -1);

  thrust::device_vector<int> d_num_items_vec(1, actual_num_items);
  const int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortPairsDescending(
      d_temp_storage,
      temp_storage_bytes,
      thrust::raw_pointer_cast(d_keys_in.data()),
      thrust::raw_pointer_cast(d_keys_out.data()),
      thrust::raw_pointer_cast(d_values_in.data()),
      thrust::raw_pointer_cast(d_values_out.data()),
      d_num_items,
      max_num_items,
      0,
      static_cast<int>(sizeof(int) * 8)));

  thrust::device_vector<std::uint8_t> d_temp(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(d_temp.data());

  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortPairsDescending(
      d_temp_storage,
      temp_storage_bytes,
      thrust::raw_pointer_cast(d_keys_in.data()),
      thrust::raw_pointer_cast(d_keys_out.data()),
      thrust::raw_pointer_cast(d_values_in.data()),
      thrust::raw_pointer_cast(d_values_out.data()),
      d_num_items,
      max_num_items,
      0,
      static_cast<int>(sizeof(int) * 8)));

  std::vector<int> ref_keys, ref_values;
  sort_pairs_reference(h_keys, h_values, ref_keys, ref_values, actual_num_items, true);

  thrust::host_vector<int> h_keys_out(d_keys_out);
  thrust::host_vector<int> h_values_out(d_values_out);
  for (int i = 0; i < actual_num_items; ++i)
  {
    REQUIRE(h_keys_out[i] == ref_keys[i]);
    REQUIRE(h_values_out[i] == ref_values[i]);
  }
}

TEST_CASE("DeviceRadixSort::SortKeys indirect num_items", "[radix_sort][device]")
{
  constexpr int max_num_items = 1000;
  const int actual_num_items  = GENERATE(0, 1, 100, 1000);

  std::vector<int> h_keys(max_num_items);
  for (int i = 0; i < max_num_items; ++i)
  {
    h_keys[i] = max_num_items - i;
  }
  thrust::device_vector<int> d_keys_in(h_keys.begin(), h_keys.end());
  thrust::device_vector<int> d_keys_out(max_num_items, -1);

  thrust::device_vector<int> d_num_items_vec(1, actual_num_items);
  const int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortKeys(
      d_temp_storage,
      temp_storage_bytes,
      thrust::raw_pointer_cast(d_keys_in.data()),
      thrust::raw_pointer_cast(d_keys_out.data()),
      d_num_items,
      max_num_items,
      0,
      static_cast<int>(sizeof(int) * 8)));

  thrust::device_vector<std::uint8_t> d_temp(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(d_temp.data());

  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortKeys(
      d_temp_storage,
      temp_storage_bytes,
      thrust::raw_pointer_cast(d_keys_in.data()),
      thrust::raw_pointer_cast(d_keys_out.data()),
      d_num_items,
      max_num_items,
      0,
      static_cast<int>(sizeof(int) * 8)));

  auto ref_keys = sort_reference(h_keys, actual_num_items, false);

  thrust::host_vector<int> h_keys_out(d_keys_out);
  for (int i = 0; i < actual_num_items; ++i)
  {
    REQUIRE(h_keys_out[i] == ref_keys[i]);
  }
}

TEST_CASE("DeviceRadixSort::SortKeysDescending indirect num_items", "[radix_sort][device]")
{
  constexpr int max_num_items = 1000;
  const int actual_num_items  = GENERATE(0, 1, 100, 1000);

  std::vector<int> h_keys(max_num_items);
  for (int i = 0; i < max_num_items; ++i)
  {
    h_keys[i] = i;
  }
  thrust::device_vector<int> d_keys_in(h_keys.begin(), h_keys.end());
  thrust::device_vector<int> d_keys_out(max_num_items, -1);

  thrust::device_vector<int> d_num_items_vec(1, actual_num_items);
  const int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortKeysDescending(
      d_temp_storage,
      temp_storage_bytes,
      thrust::raw_pointer_cast(d_keys_in.data()),
      thrust::raw_pointer_cast(d_keys_out.data()),
      d_num_items,
      max_num_items,
      0,
      static_cast<int>(sizeof(int) * 8)));

  thrust::device_vector<std::uint8_t> d_temp(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(d_temp.data());

  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortKeysDescending(
      d_temp_storage,
      temp_storage_bytes,
      thrust::raw_pointer_cast(d_keys_in.data()),
      thrust::raw_pointer_cast(d_keys_out.data()),
      d_num_items,
      max_num_items,
      0,
      static_cast<int>(sizeof(int) * 8)));

  auto ref_keys = sort_reference(h_keys, actual_num_items, true);

  thrust::host_vector<int> h_keys_out(d_keys_out);
  for (int i = 0; i < actual_num_items; ++i)
  {
    REQUIRE(h_keys_out[i] == ref_keys[i]);
  }
}

TEST_CASE("DeviceRadixSort::SortPairs indirect with float keys", "[radix_sort][device]")
{
  constexpr int max_num_items = 500;
  const int actual_num_items  = 200;

  std::vector<float> h_keys(max_num_items);
  std::vector<int> h_values(max_num_items);
  for (int i = 0; i < max_num_items; ++i)
  {
    h_keys[i]   = static_cast<float>(max_num_items - i);
    h_values[i] = i;
  }
  thrust::device_vector<float> d_keys_in(h_keys.begin(), h_keys.end());
  thrust::device_vector<float> d_keys_out(max_num_items, -1.0f);
  thrust::device_vector<int> d_values_in(h_values.begin(), h_values.end());
  thrust::device_vector<int> d_values_out(max_num_items, -1);

  thrust::device_vector<int> d_num_items_vec(1, actual_num_items);
  const int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

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
      0,
      static_cast<int>(sizeof(int) * 8)));

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
      0,
      static_cast<int>(sizeof(int) * 8)));

  std::vector<float> ref_keys;
  std::vector<int> ref_values;
  sort_pairs_reference(h_keys, h_values, ref_keys, ref_values, actual_num_items, false);

  thrust::host_vector<float> h_keys_out(d_keys_out);
  thrust::host_vector<int> h_values_out(d_values_out);
  for (int i = 0; i < actual_num_items; ++i)
  {
    REQUIRE(h_keys_out[i] == ref_keys[i]);
    REQUIRE(h_values_out[i] == ref_values[i]);
  }
}

TEST_CASE("DeviceRadixSort indirect num_items varying between calls", "[radix_sort][device]")
{
  constexpr int max_num_items = 2000;

  std::vector<int> h_keys(max_num_items);
  std::vector<int> h_values(max_num_items);
  for (int i = 0; i < max_num_items; ++i)
  {
    h_keys[i]   = max_num_items - i;
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
      static_cast<int>(sizeof(int) * 8)));

  thrust::device_vector<std::uint8_t> d_temp(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(d_temp.data());

  for (int n : {10, 500, 1, 2000, 0, 42})
  {
    d_num_items_vec[0] = n;
    thrust::fill(d_keys_out.begin(), d_keys_out.end(), -1);
    thrust::fill(d_values_out.begin(), d_values_out.end(), -1);

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
        static_cast<int>(sizeof(int) * 8)));

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
