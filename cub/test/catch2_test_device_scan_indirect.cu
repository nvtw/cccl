// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_scan.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <numeric>

#include <c2h/catch2_test_helper.h>

// Helper: compute exclusive sum reference on host
template <typename T>
std::vector<T> exclusive_sum_reference(const std::vector<T>& input, int num_items)
{
  std::vector<T> result(num_items);
  T acc{};
  for (int i = 0; i < num_items; ++i)
  {
    result[i] = acc;
    acc += input[i];
  }
  return result;
}

// Helper: compute inclusive sum reference on host
template <typename T>
std::vector<T> inclusive_sum_reference(const std::vector<T>& input, int num_items)
{
  std::vector<T> result(num_items);
  T acc{};
  for (int i = 0; i < num_items; ++i)
  {
    acc += input[i];
    result[i] = acc;
  }
  return result;
}

TEST_CASE("DeviceScan::ExclusiveSum indirect num_items", "[scan][device]")
{
  constexpr int max_num_items = 1000;
  const int actual_num_items  = GENERATE(0, 1, 42, 500, 1000);

  // Prepare input
  std::vector<int> h_in(max_num_items);
  std::iota(h_in.begin(), h_in.end(), 1);
  thrust::device_vector<int> d_in(h_in.begin(), h_in.end());
  thrust::device_vector<int> d_out(max_num_items, -1);

  // Put actual num_items in device memory
  thrust::device_vector<int> d_num_items_vec(1, actual_num_items);
  const int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

  // Query temp storage
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  REQUIRE(cudaSuccess
          == cub::DeviceScan::ExclusiveSum(
            d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), d_num_items, max_num_items));
  REQUIRE(temp_storage_bytes > 0);

  // Allocate temp storage
  thrust::device_vector<std::uint8_t> d_temp(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(d_temp.data());

  // Run
  REQUIRE(cudaSuccess
          == cub::DeviceScan::ExclusiveSum(
            d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), d_num_items, max_num_items));

  // Verify
  auto h_ref = exclusive_sum_reference(h_in, actual_num_items);
  thrust::host_vector<int> h_out(d_out);
  for (int i = 0; i < actual_num_items; ++i)
  {
    REQUIRE(h_out[i] == h_ref[i]);
  }
}

TEST_CASE("DeviceScan::InclusiveSum indirect num_items", "[scan][device]")
{
  constexpr int max_num_items = 1000;
  const int actual_num_items  = GENERATE(0, 1, 42, 500, 1000);

  std::vector<int> h_in(max_num_items);
  std::iota(h_in.begin(), h_in.end(), 1);
  thrust::device_vector<int> d_in(h_in.begin(), h_in.end());
  thrust::device_vector<int> d_out(max_num_items, -1);

  thrust::device_vector<int> d_num_items_vec(1, actual_num_items);
  const int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  REQUIRE(cudaSuccess
          == cub::DeviceScan::InclusiveSum(
            d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), d_num_items, max_num_items));

  thrust::device_vector<std::uint8_t> d_temp(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(d_temp.data());

  REQUIRE(cudaSuccess
          == cub::DeviceScan::InclusiveSum(
            d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), d_num_items, max_num_items));

  auto h_ref = inclusive_sum_reference(h_in, actual_num_items);
  thrust::host_vector<int> h_out(d_out);
  for (int i = 0; i < actual_num_items; ++i)
  {
    REQUIRE(h_out[i] == h_ref[i]);
  }
}

TEST_CASE("DeviceScan::ExclusiveScan indirect num_items", "[scan][device]")
{
  constexpr int max_num_items = 512;
  const int actual_num_items  = GENERATE(0, 1, 100, 512);

  std::vector<int> h_in(max_num_items);
  std::iota(h_in.begin(), h_in.end(), 1);
  thrust::device_vector<int> d_in(h_in.begin(), h_in.end());
  thrust::device_vector<int> d_out(max_num_items, -1);

  thrust::device_vector<int> d_num_items_vec(1, actual_num_items);
  const int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

  const int init_value = 42;

  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::ExclusiveScan(
      d_temp_storage,
      temp_storage_bytes,
      d_in.begin(),
      d_out.begin(),
      ::cuda::std::plus<>{},
      init_value,
      d_num_items,
      max_num_items));

  thrust::device_vector<std::uint8_t> d_temp(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(d_temp.data());

  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::ExclusiveScan(
      d_temp_storage,
      temp_storage_bytes,
      d_in.begin(),
      d_out.begin(),
      ::cuda::std::plus<>{},
      init_value,
      d_num_items,
      max_num_items));

  // Compute reference
  std::vector<int> h_ref(actual_num_items);
  int acc = init_value;
  for (int i = 0; i < actual_num_items; ++i)
  {
    h_ref[i] = acc;
    acc += h_in[i];
  }

  thrust::host_vector<int> h_out(d_out);
  for (int i = 0; i < actual_num_items; ++i)
  {
    REQUIRE(h_out[i] == h_ref[i]);
  }
}

TEST_CASE("DeviceScan::InclusiveScan indirect num_items", "[scan][device]")
{
  constexpr int max_num_items = 512;
  const int actual_num_items  = GENERATE(0, 1, 100, 512);

  std::vector<int> h_in(max_num_items);
  std::iota(h_in.begin(), h_in.end(), 1);
  thrust::device_vector<int> d_in(h_in.begin(), h_in.end());
  thrust::device_vector<int> d_out(max_num_items, -1);

  thrust::device_vector<int> d_num_items_vec(1, actual_num_items);
  const int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::InclusiveScan(
      d_temp_storage,
      temp_storage_bytes,
      d_in.begin(),
      d_out.begin(),
      ::cuda::std::plus<>{},
      d_num_items,
      max_num_items));

  thrust::device_vector<std::uint8_t> d_temp(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(d_temp.data());

  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::InclusiveScan(
      d_temp_storage,
      temp_storage_bytes,
      d_in.begin(),
      d_out.begin(),
      ::cuda::std::plus<>{},
      d_num_items,
      max_num_items));

  auto h_ref = inclusive_sum_reference(h_in, actual_num_items);
  thrust::host_vector<int> h_out(d_out);
  for (int i = 0; i < actual_num_items; ++i)
  {
    REQUIRE(h_out[i] == h_ref[i]);
  }
}

TEST_CASE("DeviceScan indirect num_items with float", "[scan][device]")
{
  constexpr int max_num_items = 256;
  const int actual_num_items  = 100;

  std::vector<float> h_in(max_num_items, 1.0f);
  thrust::device_vector<float> d_in(h_in.begin(), h_in.end());
  thrust::device_vector<float> d_out(max_num_items, -1.0f);

  thrust::device_vector<int> d_num_items_vec(1, actual_num_items);
  const int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  REQUIRE(cudaSuccess
          == cub::DeviceScan::ExclusiveSum(
            d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), d_num_items, max_num_items));

  thrust::device_vector<std::uint8_t> d_temp(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(d_temp.data());

  REQUIRE(cudaSuccess
          == cub::DeviceScan::ExclusiveSum(
            d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), d_num_items, max_num_items));

  thrust::host_vector<float> h_out(d_out);
  for (int i = 0; i < actual_num_items; ++i)
  {
    REQUIRE(h_out[i] == static_cast<float>(i));
  }
}

TEST_CASE("DeviceScan indirect num_items varying between calls", "[scan][device]")
{
  constexpr int max_num_items = 1000;

  std::vector<int> h_in(max_num_items, 1);
  thrust::device_vector<int> d_in(h_in.begin(), h_in.end());
  thrust::device_vector<int> d_out(max_num_items);

  thrust::device_vector<int> d_num_items_vec(1, 0);
  int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

  // Allocate temp storage for max
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  REQUIRE(cudaSuccess
          == cub::DeviceScan::ExclusiveSum(
            d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), d_num_items, max_num_items));

  thrust::device_vector<std::uint8_t> d_temp(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(d_temp.data());

  // Run with different num_items values, reusing same temp storage
  for (int n : {10, 500, 1, 1000, 0, 42})
  {
    d_num_items_vec[0] = n;
    thrust::fill(d_out.begin(), d_out.end(), -1);

    REQUIRE(cudaSuccess
            == cub::DeviceScan::ExclusiveSum(
              d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), d_num_items, max_num_items));

    thrust::host_vector<int> h_out(d_out);
    for (int i = 0; i < n; ++i)
    {
      REQUIRE(h_out[i] == i);
    }
  }
}
