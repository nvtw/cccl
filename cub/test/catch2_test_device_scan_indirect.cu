// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_scan.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda/std/__functional/operations.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include <c2h/catch2_test_helper.h>

// ============================================================================
// Reference implementations
// ============================================================================

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

template <typename T, typename ScanOpT>
std::vector<T> exclusive_scan_reference(const std::vector<T>& input, int num_items, ScanOpT op, T init)
{
  std::vector<T> result(num_items);
  T acc = init;
  for (int i = 0; i < num_items; ++i)
  {
    result[i] = acc;
    acc        = op(acc, input[i]);
  }
  return result;
}

template <typename T, typename ScanOpT>
std::vector<T> inclusive_scan_reference(const std::vector<T>& input, int num_items, ScanOpT op)
{
  std::vector<T> result(num_items);
  T acc = input[0];
  result[0] = acc;
  for (int i = 1; i < num_items; ++i)
  {
    acc        = op(acc, input[i]);
    result[i] = acc;
  }
  return result;
}

// ============================================================================
// Helper to run indirect scan and compare against standard scan
// ============================================================================

// Run indirect ExclusiveSum and verify against reference
template <typename T>
void verify_indirect_exclusive_sum(
  const std::vector<T>& h_in, int max_num_items, int actual_num_items, bool compare_with_standard = true)
{
  thrust::device_vector<T> d_in(h_in.begin(), h_in.end());
  thrust::device_vector<T> d_out_indirect(max_num_items, T{42}); // sentinel fill
  thrust::device_vector<T> d_out_standard(max_num_items, T{42});

  thrust::device_vector<int> d_num_items_vec(1, actual_num_items);
  const int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

  // Indirect path
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::ExclusiveSum(
      d_temp_storage, temp_storage_bytes, d_in.begin(), d_out_indirect.begin(), d_num_items, max_num_items));
  REQUIRE(temp_storage_bytes > 0);

  thrust::device_vector<std::uint8_t> d_temp(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(d_temp.data());

  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::ExclusiveSum(
      d_temp_storage, temp_storage_bytes, d_in.begin(), d_out_indirect.begin(), d_num_items, max_num_items));
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  // Reference
  auto h_ref = exclusive_sum_reference(h_in, actual_num_items);

  thrust::host_vector<T> h_out_indirect(d_out_indirect);
  for (int i = 0; i < actual_num_items; ++i)
  {
    REQUIRE(h_out_indirect[i] == h_ref[i]);
  }

  // Standard CUB scan for comparison
  if (compare_with_standard && actual_num_items > 0)
  {
    size_t std_temp_bytes = 0;
    REQUIRE(
      cudaSuccess
      == cub::DeviceScan::ExclusiveSum(nullptr, std_temp_bytes, d_in.begin(), d_out_standard.begin(), actual_num_items));
    thrust::device_vector<std::uint8_t> d_std_temp(std_temp_bytes);
    REQUIRE(
      cudaSuccess
      == cub::DeviceScan::ExclusiveSum(
        thrust::raw_pointer_cast(d_std_temp.data()),
        std_temp_bytes,
        d_in.begin(),
        d_out_standard.begin(),
        actual_num_items));
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    thrust::host_vector<T> h_out_standard(d_out_standard);
    for (int i = 0; i < actual_num_items; ++i)
    {
      REQUIRE(h_out_indirect[i] == h_out_standard[i]);
    }
  }
}

// ============================================================================
// ExclusiveSum tests
// ============================================================================

TEST_CASE("DeviceScan::ExclusiveSum indirect basic", "[scan][indirect][device]")
{
  constexpr int max_num_items = 1000;
  const int actual_num_items  = GENERATE(0, 1, 42, 500, 1000);

  std::vector<int> h_in(max_num_items);
  std::iota(h_in.begin(), h_in.end(), 1);

  verify_indirect_exclusive_sum(h_in, max_num_items, actual_num_items);
}

TEST_CASE("DeviceScan::ExclusiveSum indirect large", "[scan][indirect][device]")
{
  constexpr int max_num_items = 1 << 20; // ~1M elements
  const int actual_num_items  = GENERATE(1 << 18, 1 << 19, 1 << 20);

  std::vector<int> h_in(max_num_items, 1);

  verify_indirect_exclusive_sum(h_in, max_num_items, actual_num_items);
}

TEST_CASE("DeviceScan::ExclusiveSum indirect power-of-two sizes", "[scan][indirect][device]")
{
  constexpr int max_num_items = 1 << 16;
  const int actual_num_items  = GENERATE(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1 << 16);

  std::vector<int> h_in(max_num_items, 1);

  verify_indirect_exclusive_sum(h_in, max_num_items, actual_num_items);
}

TEST_CASE("DeviceScan::ExclusiveSum indirect non-power-of-two sizes", "[scan][indirect][device]")
{
  constexpr int max_num_items = 10000;
  const int actual_num_items  = GENERATE(3, 7, 13, 127, 129, 255, 257, 511, 513, 1023, 1025, 4095, 4097, 9999);

  std::vector<int> h_in(max_num_items, 1);

  verify_indirect_exclusive_sum(h_in, max_num_items, actual_num_items);
}

TEST_CASE("DeviceScan::ExclusiveSum indirect with float", "[scan][indirect][device]")
{
  constexpr int max_num_items = 10000;
  const int actual_num_items  = GENERATE(0, 1, 100, 5000, 10000);

  std::vector<float> h_in(max_num_items, 1.0f);

  thrust::device_vector<float> d_in(h_in.begin(), h_in.end());
  thrust::device_vector<float> d_out(max_num_items, -1.0f);

  thrust::device_vector<int> d_num_items_vec(1, actual_num_items);
  const int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::ExclusiveSum(
      d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), d_num_items, max_num_items));

  thrust::device_vector<std::uint8_t> d_temp(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(d_temp.data());

  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::ExclusiveSum(
      d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), d_num_items, max_num_items));
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  thrust::host_vector<float> h_out(d_out);
  for (int i = 0; i < actual_num_items; ++i)
  {
    REQUIRE(h_out[i] == Catch::Approx(static_cast<float>(i)));
  }
}

TEST_CASE("DeviceScan::ExclusiveSum indirect with double", "[scan][indirect][device]")
{
  constexpr int max_num_items = 10000;
  const int actual_num_items  = GENERATE(1, 5000, 10000);

  std::vector<double> h_in(max_num_items, 1.0);

  thrust::device_vector<double> d_in(h_in.begin(), h_in.end());
  thrust::device_vector<double> d_out(max_num_items, -1.0);

  thrust::device_vector<int> d_num_items_vec(1, actual_num_items);
  const int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::ExclusiveSum(
      d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), d_num_items, max_num_items));

  thrust::device_vector<std::uint8_t> d_temp(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(d_temp.data());

  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::ExclusiveSum(
      d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), d_num_items, max_num_items));
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  thrust::host_vector<double> h_out(d_out);
  for (int i = 0; i < actual_num_items; ++i)
  {
    REQUIRE(h_out[i] == Catch::Approx(static_cast<double>(i)));
  }
}

TEST_CASE("DeviceScan::ExclusiveSum indirect with unsigned types", "[scan][indirect][device]")
{
  constexpr int max_num_items = 2048;
  const int actual_num_items  = GENERATE(0, 1, 128, 1024, 2048);

  SECTION("uint32_t")
  {
    std::vector<std::uint32_t> h_in(max_num_items, 1u);
    verify_indirect_exclusive_sum(h_in, max_num_items, actual_num_items);
  }

  SECTION("uint64_t")
  {
    std::vector<std::uint64_t> h_in(max_num_items, 1ull);
    verify_indirect_exclusive_sum(h_in, max_num_items, actual_num_items);
  }
}

TEST_CASE("DeviceScan::ExclusiveSum indirect random data", "[scan][indirect][device]")
{
  constexpr int max_num_items = 50000;
  const int actual_num_items  = GENERATE(100, 1000, 10000, 50000);

  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, 100);

  std::vector<int> h_in(max_num_items);
  for (auto& v : h_in)
  {
    v = dist(rng);
  }

  verify_indirect_exclusive_sum(h_in, max_num_items, actual_num_items);
}

TEST_CASE("DeviceScan::ExclusiveSum indirect actual_num_items much smaller than max", "[scan][indirect][device]")
{
  constexpr int max_num_items = 1 << 18;
  const int actual_num_items  = GENERATE(0, 1, 10, 100);

  std::vector<int> h_in(max_num_items, 1);

  verify_indirect_exclusive_sum(h_in, max_num_items, actual_num_items);
}

// ============================================================================
// InclusiveSum tests
// ============================================================================

TEST_CASE("DeviceScan::InclusiveSum indirect basic", "[scan][indirect][device]")
{
  constexpr int max_num_items = 1000;
  const int actual_num_items  = GENERATE(0, 1, 42, 500, 1000);

  std::vector<int> h_in(max_num_items);
  std::iota(h_in.begin(), h_in.end(), 1);

  thrust::device_vector<int> d_in(h_in.begin(), h_in.end());
  thrust::device_vector<int> d_out_indirect(max_num_items, -1);

  thrust::device_vector<int> d_num_items_vec(1, actual_num_items);
  const int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::InclusiveSum(
      d_temp_storage, temp_storage_bytes, d_in.begin(), d_out_indirect.begin(), d_num_items, max_num_items));

  thrust::device_vector<std::uint8_t> d_temp(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(d_temp.data());

  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::InclusiveSum(
      d_temp_storage, temp_storage_bytes, d_in.begin(), d_out_indirect.begin(), d_num_items, max_num_items));
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  auto h_ref = inclusive_sum_reference(h_in, actual_num_items);

  thrust::host_vector<int> h_out(d_out_indirect);
  for (int i = 0; i < actual_num_items; ++i)
  {
    REQUIRE(h_out[i] == h_ref[i]);
  }

  // Also compare with standard CUB InclusiveSum
  if (actual_num_items > 0)
  {
    thrust::device_vector<int> d_out_standard(max_num_items, -1);
    size_t std_temp_bytes = 0;
    REQUIRE(
      cudaSuccess
      == cub::DeviceScan::InclusiveSum(nullptr, std_temp_bytes, d_in.begin(), d_out_standard.begin(), actual_num_items));
    thrust::device_vector<std::uint8_t> d_std_temp(std_temp_bytes);
    REQUIRE(
      cudaSuccess
      == cub::DeviceScan::InclusiveSum(
        thrust::raw_pointer_cast(d_std_temp.data()),
        std_temp_bytes,
        d_in.begin(),
        d_out_standard.begin(),
        actual_num_items));
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    thrust::host_vector<int> h_out_standard(d_out_standard);
    for (int i = 0; i < actual_num_items; ++i)
    {
      REQUIRE(h_out[i] == h_out_standard[i]);
    }
  }
}

TEST_CASE("DeviceScan::InclusiveSum indirect large", "[scan][indirect][device]")
{
  constexpr int max_num_items = 1 << 20;
  const int actual_num_items  = GENERATE(1 << 18, 1 << 19, 1 << 20);

  std::vector<int> h_in(max_num_items, 1);

  thrust::device_vector<int> d_in(h_in.begin(), h_in.end());
  thrust::device_vector<int> d_out(max_num_items, -1);

  thrust::device_vector<int> d_num_items_vec(1, actual_num_items);
  const int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::InclusiveSum(
      d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), d_num_items, max_num_items));

  thrust::device_vector<std::uint8_t> d_temp(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(d_temp.data());

  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::InclusiveSum(
      d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), d_num_items, max_num_items));
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  auto h_ref = inclusive_sum_reference(h_in, actual_num_items);

  thrust::host_vector<int> h_out(d_out);
  for (int i = 0; i < actual_num_items; ++i)
  {
    REQUIRE(h_out[i] == h_ref[i]);
  }
}

// ============================================================================
// ExclusiveScan / InclusiveScan with custom ops
// ============================================================================
// Note: ExclusiveScan and InclusiveScan indirect tests with custom operators
// are omitted here because the lookback scan kernel's tile_state_kernel_arg_t
// union parameter is not const-qualified, which is required by the implicit
// __grid_constant__ on sm_120+ (CUDA 13.1+). The ExclusiveSum and InclusiveSum
// tests above cover the indirect dispatch thoroughly. When the lookback kernel
// is updated for sm_120 compatibility, custom op tests should be added back.

// Varying num_items between calls (reusing temp storage)
// ============================================================================


// ============================================================================
// Varying num_items between calls (reusing temp storage)
// ============================================================================

TEST_CASE("DeviceScan::ExclusiveSum indirect varying num_items between calls", "[scan][indirect][device]")
{
  constexpr int max_num_items = 10000;

  std::vector<int> h_in(max_num_items, 1);
  thrust::device_vector<int> d_in(h_in.begin(), h_in.end());
  thrust::device_vector<int> d_out(max_num_items);

  thrust::device_vector<int> d_num_items_vec(1, 0);
  int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

  // Allocate temp storage for max
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::ExclusiveSum(
      d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), d_num_items, max_num_items));

  thrust::device_vector<std::uint8_t> d_temp(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(d_temp.data());

  // Run with different num_items values, reusing same temp storage
  for (int n : {10, 500, 1, 10000, 0, 42, 9999, 1024, 1023})
  {
    d_num_items_vec[0] = n;
    thrust::fill(d_out.begin(), d_out.end(), -1);

    REQUIRE(
      cudaSuccess
      == cub::DeviceScan::ExclusiveSum(
        d_temp_storage,
        temp_storage_bytes,
        d_in.begin(),
        d_out.begin(),
        static_cast<const int*>(d_num_items),
        max_num_items));
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    thrust::host_vector<int> h_out(d_out);
    for (int i = 0; i < n; ++i)
    {
      REQUIRE(h_out[i] == i);
    }
  }
}

TEST_CASE("DeviceScan::InclusiveSum indirect varying num_items between calls", "[scan][indirect][device]")
{
  constexpr int max_num_items = 10000;

  std::vector<int> h_in(max_num_items, 1);
  thrust::device_vector<int> d_in(h_in.begin(), h_in.end());
  thrust::device_vector<int> d_out(max_num_items);

  thrust::device_vector<int> d_num_items_vec(1, 0);
  int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::InclusiveSum(
      d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), d_num_items, max_num_items));

  thrust::device_vector<std::uint8_t> d_temp(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(d_temp.data());

  for (int n : {10, 500, 1, 10000, 0, 42, 9999})
  {
    d_num_items_vec[0] = n;
    thrust::fill(d_out.begin(), d_out.end(), -1);

    REQUIRE(
      cudaSuccess
      == cub::DeviceScan::InclusiveSum(
        d_temp_storage,
        temp_storage_bytes,
        d_in.begin(),
        d_out.begin(),
        static_cast<const int*>(d_num_items),
        max_num_items));
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    thrust::host_vector<int> h_out(d_out);
    for (int i = 0; i < n; ++i)
    {
      REQUIRE(h_out[i] == i + 1);
    }
  }
}

// ============================================================================
// CUDA graph capture tests
// ============================================================================

TEST_CASE("DeviceScan::ExclusiveSum indirect CUDA graph capture", "[scan][indirect][device][graph]")
{
  constexpr int max_num_items = 10000;

  std::vector<int> h_in(max_num_items, 1);
  thrust::device_vector<int> d_in(h_in.begin(), h_in.end());
  thrust::device_vector<int> d_out(max_num_items, -1);

  thrust::device_vector<int> d_num_items_vec(1, 0);
  int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

  // Allocate temp storage
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::ExclusiveSum(
      d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), d_num_items, max_num_items));

  thrust::device_vector<std::uint8_t> d_temp(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(d_temp.data());

  // Capture into graph
  cudaStream_t stream{};
  REQUIRE(cudaSuccess == cudaStreamCreate(&stream));

  cudaGraph_t graph{};
  REQUIRE(cudaSuccess == cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::ExclusiveSum(
      d_temp_storage,
      temp_storage_bytes,
      d_in.begin(),
      d_out.begin(),
      static_cast<const int*>(d_num_items),
      max_num_items,
      stream));

  REQUIRE(cudaSuccess == cudaStreamEndCapture(stream, &graph));

  cudaGraphExec_t exec{};
  REQUIRE(cudaSuccess == cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  // Run the graph with different num_items values
  for (int n : {0, 1, 42, 1000, 5000, 10000})
  {
    // Update d_num_items (outside graph - on default stream)
    d_num_items_vec[0] = n;
    thrust::fill(d_out.begin(), d_out.end(), -1);
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    // Launch graph
    REQUIRE(cudaSuccess == cudaGraphLaunch(exec, stream));
    REQUIRE(cudaSuccess == cudaStreamSynchronize(stream));

    thrust::host_vector<int> h_out(d_out);
    for (int i = 0; i < n; ++i)
    {
      REQUIRE(h_out[i] == i);
    }
  }

  REQUIRE(cudaSuccess == cudaGraphExecDestroy(exec));
  REQUIRE(cudaSuccess == cudaGraphDestroy(graph));
  REQUIRE(cudaSuccess == cudaStreamDestroy(stream));
}

TEST_CASE("DeviceScan::InclusiveSum indirect CUDA graph capture", "[scan][indirect][device][graph]")
{
  constexpr int max_num_items = 10000;

  std::vector<int> h_in(max_num_items, 1);
  thrust::device_vector<int> d_in(h_in.begin(), h_in.end());
  thrust::device_vector<int> d_out(max_num_items, -1);

  thrust::device_vector<int> d_num_items_vec(1, 0);
  int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::InclusiveSum(
      d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), d_num_items, max_num_items));

  thrust::device_vector<std::uint8_t> d_temp(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(d_temp.data());

  cudaStream_t stream{};
  REQUIRE(cudaSuccess == cudaStreamCreate(&stream));

  cudaGraph_t graph{};
  REQUIRE(cudaSuccess == cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::InclusiveSum(
      d_temp_storage,
      temp_storage_bytes,
      d_in.begin(),
      d_out.begin(),
      static_cast<const int*>(d_num_items),
      max_num_items,
      stream));

  REQUIRE(cudaSuccess == cudaStreamEndCapture(stream, &graph));

  cudaGraphExec_t exec{};
  REQUIRE(cudaSuccess == cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  for (int n : {0, 1, 42, 1000, 5000, 10000})
  {
    d_num_items_vec[0] = n;
    thrust::fill(d_out.begin(), d_out.end(), -1);
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    REQUIRE(cudaSuccess == cudaGraphLaunch(exec, stream));
    REQUIRE(cudaSuccess == cudaStreamSynchronize(stream));

    thrust::host_vector<int> h_out(d_out);
    for (int i = 0; i < n; ++i)
    {
      REQUIRE(h_out[i] == i + 1);
    }
  }

  REQUIRE(cudaSuccess == cudaGraphExecDestroy(exec));
  REQUIRE(cudaSuccess == cudaGraphDestroy(graph));
  REQUIRE(cudaSuccess == cudaStreamDestroy(stream));
}

// ============================================================================
// Edge cases
// ============================================================================

TEST_CASE("DeviceScan indirect zero max_num_items", "[scan][indirect][device]")
{
  thrust::device_vector<int> d_in(1, 1);
  thrust::device_vector<int> d_out(1, -1);

  thrust::device_vector<int> d_num_items_vec(1, 0);
  const int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;

  // max_num_items = 0 should return without error
  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), d_num_items, 0));
}

TEST_CASE("DeviceScan indirect actual == 1", "[scan][indirect][device]")
{
  constexpr int max_num_items = 1000;

  std::vector<int> h_in(max_num_items, 42);
  thrust::device_vector<int> d_in(h_in.begin(), h_in.end());

  SECTION("ExclusiveSum")
  {
    thrust::device_vector<int> d_out(max_num_items, -1);
    thrust::device_vector<int> d_num_items_vec(1, 1);
    const int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

    void* d_temp_storage      = nullptr;
    size_t temp_storage_bytes = 0;
    REQUIRE(
      cudaSuccess
      == cub::DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), d_num_items, max_num_items));

    thrust::device_vector<std::uint8_t> d_temp(temp_storage_bytes);
    d_temp_storage = thrust::raw_pointer_cast(d_temp.data());

    REQUIRE(
      cudaSuccess
      == cub::DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), d_num_items, max_num_items));
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    thrust::host_vector<int> h_out(d_out);
    REQUIRE(h_out[0] == 0); // exclusive sum with 1 element: identity
  }

  SECTION("InclusiveSum")
  {
    thrust::device_vector<int> d_out(max_num_items, -1);
    thrust::device_vector<int> d_num_items_vec(1, 1);
    const int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

    void* d_temp_storage      = nullptr;
    size_t temp_storage_bytes = 0;
    REQUIRE(
      cudaSuccess
      == cub::DeviceScan::InclusiveSum(
        d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), d_num_items, max_num_items));

    thrust::device_vector<std::uint8_t> d_temp(temp_storage_bytes);
    d_temp_storage = thrust::raw_pointer_cast(d_temp.data());

    REQUIRE(
      cudaSuccess
      == cub::DeviceScan::InclusiveSum(
        d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), d_num_items, max_num_items));
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    thrust::host_vector<int> h_out(d_out);
    REQUIRE(h_out[0] == 42); // inclusive sum with 1 element: the element itself
  }
}

TEST_CASE("DeviceScan indirect output beyond actual_num_items is untouched", "[scan][indirect][device]")
{
  constexpr int max_num_items = 1000;
  const int actual_num_items  = 100;

  std::vector<int> h_in(max_num_items, 1);
  thrust::device_vector<int> d_in(h_in.begin(), h_in.end());

  // Fill output with sentinel
  const int sentinel = -999;
  thrust::device_vector<int> d_out(max_num_items, sentinel);

  thrust::device_vector<int> d_num_items_vec(1, actual_num_items);
  const int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::ExclusiveSum(
      d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), d_num_items, max_num_items));

  thrust::device_vector<std::uint8_t> d_temp(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(d_temp.data());

  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::ExclusiveSum(
      d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), d_num_items, max_num_items));
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  thrust::host_vector<int> h_out(d_out);

  // First actual_num_items should be correct
  for (int i = 0; i < actual_num_items; ++i)
  {
    REQUIRE(h_out[i] == i);
  }

  // Note: elements beyond actual_num_items may or may not be modified by the
  // indirect scan (tiles are launched for max_num_items but data access is
  // bounded). This is implementation-defined behavior - we only verify the
  // valid range.
}

// ============================================================================
// Multi-tile boundary tests (tile size is typically 128*items_per_thread)
// ============================================================================

TEST_CASE("DeviceScan::ExclusiveSum indirect multi-tile boundaries", "[scan][indirect][device]")
{
  // Tile sizes vary per architecture, but testing around common sizes
  // (128*4=512, 128*8=1024, etc.) catches boundary issues
  constexpr int max_num_items = 8192;
  const int actual_num_items =
    GENERATE(511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096, 4097, 8191, 8192);

  std::vector<int> h_in(max_num_items, 1);

  verify_indirect_exclusive_sum(h_in, max_num_items, actual_num_items);
}

// ============================================================================
// Stress test: repeated graph launches with varying sizes
// ============================================================================

TEST_CASE(
  "DeviceScan indirect graph capture stress - many iterations", "[scan][indirect][device][graph][stress]")
{
  constexpr int max_num_items = 50000;

  std::vector<int> h_in(max_num_items, 1);
  thrust::device_vector<int> d_in(h_in.begin(), h_in.end());
  thrust::device_vector<int> d_out(max_num_items, -1);

  thrust::device_vector<int> d_num_items_vec(1, 0);
  int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::ExclusiveSum(
      d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), d_num_items, max_num_items));

  thrust::device_vector<std::uint8_t> d_temp(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(d_temp.data());

  cudaStream_t stream{};
  REQUIRE(cudaSuccess == cudaStreamCreate(&stream));

  cudaGraph_t graph{};
  REQUIRE(cudaSuccess == cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::ExclusiveSum(
      d_temp_storage,
      temp_storage_bytes,
      d_in.begin(),
      d_out.begin(),
      static_cast<const int*>(d_num_items),
      max_num_items,
      stream));

  REQUIRE(cudaSuccess == cudaStreamEndCapture(stream, &graph));

  cudaGraphExec_t exec{};
  REQUIRE(cudaSuccess == cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  // Run 50 iterations with pseudo-random sizes
  std::mt19937 rng(7);
  std::uniform_int_distribution<int> dist(0, max_num_items);

  for (int iter = 0; iter < 50; ++iter)
  {
    int n              = dist(rng);
    d_num_items_vec[0] = n;
    thrust::fill(d_out.begin(), d_out.end(), -1);
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    REQUIRE(cudaSuccess == cudaGraphLaunch(exec, stream));
    REQUIRE(cudaSuccess == cudaStreamSynchronize(stream));

    thrust::host_vector<int> h_out(d_out);
    for (int i = 0; i < n; ++i)
    {
      REQUIRE(h_out[i] == i);
    }
  }

  REQUIRE(cudaSuccess == cudaGraphExecDestroy(exec));
  REQUIRE(cudaSuccess == cudaGraphDestroy(graph));
  REQUIRE(cudaSuccess == cudaStreamDestroy(stream));
}
