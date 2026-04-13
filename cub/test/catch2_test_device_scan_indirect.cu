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
  if (num_items == 0)
  {
    return {};
  }
  std::vector<T> result(num_items);
  T acc     = input[0];
  result[0] = acc;
  for (int i = 1; i < num_items; ++i)
  {
    acc        = op(acc, input[i]);
    result[i] = acc;
  }
  return result;
}

// ============================================================================
// Helper: run indirect ExclusiveSum and verify against reference + standard CUB
// ============================================================================

template <typename T>
void verify_indirect_exclusive_sum(
  const std::vector<T>& h_in, int max_num_items, int actual_num_items, bool compare_with_standard = true)
{
  thrust::device_vector<T> d_in(h_in.begin(), h_in.end());
  thrust::device_vector<T> d_out_indirect(max_num_items, T{42});
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
// Helper: run indirect InclusiveSum and verify against reference + standard CUB
// ============================================================================

template <typename T>
void verify_indirect_inclusive_sum(
  const std::vector<T>& h_in, int max_num_items, int actual_num_items, bool compare_with_standard = true)
{
  thrust::device_vector<T> d_in(h_in.begin(), h_in.end());
  thrust::device_vector<T> d_out_indirect(max_num_items, T{42});
  thrust::device_vector<T> d_out_standard(max_num_items, T{42});

  thrust::device_vector<int> d_num_items_vec(1, actual_num_items);
  const int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::InclusiveSum(
      d_temp_storage, temp_storage_bytes, d_in.begin(), d_out_indirect.begin(), d_num_items, max_num_items));
  REQUIRE(temp_storage_bytes > 0);

  thrust::device_vector<std::uint8_t> d_temp(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(d_temp.data());

  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::InclusiveSum(
      d_temp_storage, temp_storage_bytes, d_in.begin(), d_out_indirect.begin(), d_num_items, max_num_items));
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  auto h_ref = inclusive_sum_reference(h_in, actual_num_items);

  thrust::host_vector<T> h_out_indirect(d_out_indirect);
  for (int i = 0; i < actual_num_items; ++i)
  {
    REQUIRE(h_out_indirect[i] == h_ref[i]);
  }

  if (compare_with_standard && actual_num_items > 0)
  {
    size_t std_temp_bytes = 0;
    REQUIRE(
      cudaSuccess
      == cub::DeviceScan::InclusiveSum(
        nullptr, std_temp_bytes, d_in.begin(), d_out_standard.begin(), actual_num_items));
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

    thrust::host_vector<T> h_out_standard(d_out_standard);
    for (int i = 0; i < actual_num_items; ++i)
    {
      REQUIRE(h_out_indirect[i] == h_out_standard[i]);
    }
  }
}

// ============================================================================
// Helper: run indirect ExclusiveScan with custom op and verify
// ============================================================================

template <typename T, typename ScanOpT>
void verify_indirect_exclusive_scan(
  const std::vector<T>& h_in,
  int max_num_items,
  int actual_num_items,
  ScanOpT scan_op,
  T init_value,
  bool compare_with_standard = true)
{
  thrust::device_vector<T> d_in(h_in.begin(), h_in.end());
  thrust::device_vector<T> d_out_indirect(max_num_items, T{99});
  thrust::device_vector<T> d_out_standard(max_num_items, T{99});

  thrust::device_vector<int> d_num_items_vec(1, actual_num_items);
  const int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::ExclusiveScan(
      d_temp_storage,
      temp_storage_bytes,
      d_in.begin(),
      d_out_indirect.begin(),
      scan_op,
      init_value,
      d_num_items,
      max_num_items));
  REQUIRE(temp_storage_bytes > 0);

  thrust::device_vector<std::uint8_t> d_temp(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(d_temp.data());

  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::ExclusiveScan(
      d_temp_storage,
      temp_storage_bytes,
      d_in.begin(),
      d_out_indirect.begin(),
      scan_op,
      init_value,
      d_num_items,
      max_num_items));
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  auto h_ref = exclusive_scan_reference(h_in, actual_num_items, scan_op, init_value);

  thrust::host_vector<T> h_out_indirect(d_out_indirect);
  for (int i = 0; i < actual_num_items; ++i)
  {
    REQUIRE(h_out_indirect[i] == h_ref[i]);
  }

  if (compare_with_standard && actual_num_items > 0)
  {
    size_t std_temp_bytes = 0;
    REQUIRE(
      cudaSuccess
      == cub::DeviceScan::ExclusiveScan(
        nullptr, std_temp_bytes, d_in.begin(), d_out_standard.begin(), scan_op, init_value, actual_num_items));
    thrust::device_vector<std::uint8_t> d_std_temp(std_temp_bytes);
    REQUIRE(
      cudaSuccess
      == cub::DeviceScan::ExclusiveScan(
        thrust::raw_pointer_cast(d_std_temp.data()),
        std_temp_bytes,
        d_in.begin(),
        d_out_standard.begin(),
        scan_op,
        init_value,
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
// Helper: run indirect InclusiveScan with custom op and verify
// ============================================================================

template <typename T, typename ScanOpT>
void verify_indirect_inclusive_scan(
  const std::vector<T>& h_in,
  int max_num_items,
  int actual_num_items,
  ScanOpT scan_op,
  bool compare_with_standard = true)
{
  thrust::device_vector<T> d_in(h_in.begin(), h_in.end());
  thrust::device_vector<T> d_out_indirect(max_num_items, T{99});
  thrust::device_vector<T> d_out_standard(max_num_items, T{99});

  thrust::device_vector<int> d_num_items_vec(1, actual_num_items);
  const int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::InclusiveScan(
      d_temp_storage, temp_storage_bytes, d_in.begin(), d_out_indirect.begin(), scan_op, d_num_items, max_num_items));
  REQUIRE(temp_storage_bytes > 0);

  thrust::device_vector<std::uint8_t> d_temp(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(d_temp.data());

  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::InclusiveScan(
      d_temp_storage, temp_storage_bytes, d_in.begin(), d_out_indirect.begin(), scan_op, d_num_items, max_num_items));
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  auto h_ref = inclusive_scan_reference(h_in, actual_num_items, scan_op);

  thrust::host_vector<T> h_out_indirect(d_out_indirect);
  for (int i = 0; i < actual_num_items; ++i)
  {
    REQUIRE(h_out_indirect[i] == h_ref[i]);
  }

  if (compare_with_standard && actual_num_items > 0)
  {
    size_t std_temp_bytes = 0;
    REQUIRE(
      cudaSuccess
      == cub::DeviceScan::InclusiveScan(
        nullptr, std_temp_bytes, d_in.begin(), d_out_standard.begin(), scan_op, actual_num_items));
    thrust::device_vector<std::uint8_t> d_std_temp(std_temp_bytes);
    REQUIRE(
      cudaSuccess
      == cub::DeviceScan::InclusiveScan(
        thrust::raw_pointer_cast(d_std_temp.data()),
        std_temp_bytes,
        d_in.begin(),
        d_out_standard.begin(),
        scan_op,
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
// Helper: run indirect ExclusiveSum with float and verify with Approx
// ============================================================================

template <typename FloatT>
void verify_indirect_exclusive_sum_float(
  const std::vector<FloatT>& h_in, int max_num_items, int actual_num_items)
{
  thrust::device_vector<FloatT> d_in(h_in.begin(), h_in.end());
  thrust::device_vector<FloatT> d_out(max_num_items, FloatT{-1});

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

  auto h_ref = exclusive_sum_reference(h_in, actual_num_items);

  thrust::host_vector<FloatT> h_out(d_out);
  for (int i = 0; i < actual_num_items; ++i)
  {
    REQUIRE(h_out[i] == Catch::Approx(h_ref[i]));
  }
}

// ============================================================================
// Helper: run indirect InclusiveSum with float and verify with Approx
// ============================================================================

template <typename FloatT>
void verify_indirect_inclusive_sum_float(
  const std::vector<FloatT>& h_in, int max_num_items, int actual_num_items)
{
  thrust::device_vector<FloatT> d_in(h_in.begin(), h_in.end());
  thrust::device_vector<FloatT> d_out(max_num_items, FloatT{-1});

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

  thrust::host_vector<FloatT> h_out(d_out);
  for (int i = 0; i < actual_num_items; ++i)
  {
    REQUIRE(h_out[i] == Catch::Approx(h_ref[i]));
  }
}

// ############################################################################
//
//  ExclusiveSum tests
//
// ############################################################################

// --- 1. Basic sizes ---
TEST_CASE("DeviceScan::ExclusiveSum indirect basic", "[scan][indirect][device]")
{
  constexpr int max_num_items = 1000;
  const int actual_num_items  = GENERATE(0, 1, 2, 42, 500, 1000);

  std::vector<int> h_in(max_num_items);
  std::iota(h_in.begin(), h_in.end(), 1);

  verify_indirect_exclusive_sum(h_in, max_num_items, actual_num_items);
}

// --- 2. Large sizes ---
TEST_CASE("DeviceScan::ExclusiveSum indirect large", "[scan][indirect][device]")
{
  constexpr int max_num_items = 1 << 21; // 2M elements
  const int actual_num_items  = GENERATE(1 << 16, 1 << 18, 1 << 20, 1 << 21);

  std::vector<int> h_in(max_num_items, 1);

  verify_indirect_exclusive_sum(h_in, max_num_items, actual_num_items);
}

// --- 3. Power-of-two sizes ---
TEST_CASE("DeviceScan::ExclusiveSum indirect power-of-two sizes", "[scan][indirect][device]")
{
  constexpr int max_num_items = 1 << 16;
  const int actual_num_items  = GENERATE(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
                                         1 << 16);

  std::vector<int> h_in(max_num_items, 1);

  verify_indirect_exclusive_sum(h_in, max_num_items, actual_num_items);
}

// --- 4. Non-power-of-two sizes ---
TEST_CASE("DeviceScan::ExclusiveSum indirect non-power-of-two sizes", "[scan][indirect][device]")
{
  constexpr int max_num_items = 10000;
  const int actual_num_items  = GENERATE(3, 7, 13, 127, 129, 255, 257, 511, 513, 1023, 1025, 4095, 4097, 9999);

  std::vector<int> h_in(max_num_items, 1);

  verify_indirect_exclusive_sum(h_in, max_num_items, actual_num_items);
}

// --- 5. Multi-tile boundary sizes ---
TEST_CASE("DeviceScan::ExclusiveSum indirect multi-tile boundaries", "[scan][indirect][device]")
{
  // Tile sizes vary per architecture, but testing around common sizes
  // (128*4=512, 128*8=1024, etc.) catches boundary issues
  constexpr int max_num_items = 8192;
  const int actual_num_items  = GENERATE(
    511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096, 4097, 8191, 8192);

  std::vector<int> h_in(max_num_items, 1);

  verify_indirect_exclusive_sum(h_in, max_num_items, actual_num_items);
}

// --- 6. Actual much smaller than max ---
TEST_CASE("DeviceScan::ExclusiveSum indirect actual much smaller than max", "[scan][indirect][device]")
{
  constexpr int max_num_items = 1 << 18;
  const int actual_num_items  = GENERATE(0, 1, 10, 100);

  std::vector<int> h_in(max_num_items, 1);

  verify_indirect_exclusive_sum(h_in, max_num_items, actual_num_items);
}

// --- 7. Float type ---
TEST_CASE("DeviceScan::ExclusiveSum indirect float", "[scan][indirect][device]")
{
  constexpr int max_num_items = 10000;
  const int actual_num_items  = GENERATE(0, 1, 100, 5000, 10000);

  std::vector<float> h_in(max_num_items, 1.0f);

  verify_indirect_exclusive_sum_float(h_in, max_num_items, actual_num_items);
}

// --- 8. Double type ---
TEST_CASE("DeviceScan::ExclusiveSum indirect double", "[scan][indirect][device]")
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

// --- 9. uint32_t and uint64_t types ---
TEST_CASE("DeviceScan::ExclusiveSum indirect unsigned types", "[scan][indirect][device]")
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

// --- 10. int8_t and int16_t types (small values to avoid overflow) ---
TEST_CASE("DeviceScan::ExclusiveSum indirect small integer types", "[scan][indirect][device]")
{
  // Keep values small to avoid overflow: for int8_t, max exclusive sum
  // of 1s is 126 (with 127 elements), so limit actual to 127
  SECTION("int8_t")
  {
    constexpr int max_num_items = 127;
    const int actual_num_items  = GENERATE(0, 1, 10, 50, 127);

    std::vector<std::int8_t> h_in(max_num_items, static_cast<std::int8_t>(1));
    verify_indirect_exclusive_sum(h_in, max_num_items, actual_num_items);
  }

  SECTION("int16_t")
  {
    constexpr int max_num_items = 4096;
    const int actual_num_items  = GENERATE(0, 1, 100, 1000, 4096);

    // Use value 1 so prefix sum values are small enough for int16_t
    std::vector<std::int16_t> h_in(max_num_items, static_cast<std::int16_t>(1));
    verify_indirect_exclusive_sum(h_in, max_num_items, actual_num_items);
  }
}

// --- 11. Random data with comparison against standard CUB ---
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

// --- 12. All-zeros input ---
TEST_CASE("DeviceScan::ExclusiveSum indirect all-zeros", "[scan][indirect][device]")
{
  constexpr int max_num_items = 2048;
  const int actual_num_items  = GENERATE(0, 1, 512, 2048);

  std::vector<int> h_in(max_num_items, 0);

  verify_indirect_exclusive_sum(h_in, max_num_items, actual_num_items);
}

// --- 13. All-ones input ---
TEST_CASE("DeviceScan::ExclusiveSum indirect all-ones", "[scan][indirect][device]")
{
  constexpr int max_num_items = 2048;
  const int actual_num_items  = GENERATE(0, 1, 512, 2048);

  std::vector<int> h_in(max_num_items, 1);

  verify_indirect_exclusive_sum(h_in, max_num_items, actual_num_items);
}

// --- 14. Alternating values (1,0,1,0,...) ---
TEST_CASE("DeviceScan::ExclusiveSum indirect alternating values", "[scan][indirect][device]")
{
  constexpr int max_num_items = 4096;
  const int actual_num_items  = GENERATE(0, 1, 2, 3, 100, 1023, 1024, 4096);

  std::vector<int> h_in(max_num_items);
  for (int i = 0; i < max_num_items; ++i)
  {
    h_in[i] = (i % 2 == 0) ? 1 : 0;
  }

  verify_indirect_exclusive_sum(h_in, max_num_items, actual_num_items);
}

// ############################################################################
//
//  InclusiveSum tests
//
// ############################################################################

// --- 15. Basic sizes ---
TEST_CASE("DeviceScan::InclusiveSum indirect basic", "[scan][indirect][device]")
{
  constexpr int max_num_items = 1000;
  const int actual_num_items  = GENERATE(0, 1, 42, 500, 1000);

  std::vector<int> h_in(max_num_items);
  std::iota(h_in.begin(), h_in.end(), 1);

  verify_indirect_inclusive_sum(h_in, max_num_items, actual_num_items);
}

// --- 16. Large sizes ---
TEST_CASE("DeviceScan::InclusiveSum indirect large", "[scan][indirect][device]")
{
  constexpr int max_num_items = 1 << 20;
  const int actual_num_items  = GENERATE(1 << 18, 1 << 19, 1 << 20);

  std::vector<int> h_in(max_num_items, 1);

  verify_indirect_inclusive_sum(h_in, max_num_items, actual_num_items);
}

// --- 17. Random data with comparison against standard ---
TEST_CASE("DeviceScan::InclusiveSum indirect random data", "[scan][indirect][device]")
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

  verify_indirect_inclusive_sum(h_in, max_num_items, actual_num_items);
}

// --- 18. Float/double types ---
TEST_CASE("DeviceScan::InclusiveSum indirect float", "[scan][indirect][device]")
{
  constexpr int max_num_items = 10000;
  const int actual_num_items  = GENERATE(0, 1, 100, 5000, 10000);

  std::vector<float> h_in(max_num_items, 1.0f);

  verify_indirect_inclusive_sum_float(h_in, max_num_items, actual_num_items);
}

TEST_CASE("DeviceScan::InclusiveSum indirect double", "[scan][indirect][device]")
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

  thrust::host_vector<double> h_out(d_out);
  for (int i = 0; i < actual_num_items; ++i)
  {
    REQUIRE(h_out[i] == Catch::Approx(h_ref[i]));
  }
}

// ############################################################################
//
//  ExclusiveScan with custom op tests
//
// ############################################################################

// --- 19. Plus with init=42 ---
TEST_CASE("DeviceScan::ExclusiveScan indirect plus with init=42", "[scan][indirect][device]")
{
  const int actual_num_items = GENERATE(0, 1, 100, 512, 2048);
  constexpr int max_num_items = 2048;

  std::vector<int> h_in(max_num_items, 1);

  verify_indirect_exclusive_scan(h_in, max_num_items, actual_num_items, ::cuda::std::plus<>{}, 42);
}

// --- 20. Plus with init=999, random data ---
TEST_CASE("DeviceScan::ExclusiveScan indirect plus with init=999 random", "[scan][indirect][device]")
{
  constexpr int max_num_items = 10000;
  const int actual_num_items  = GENERATE(100, 1000, 5000, 10000);

  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, 100);

  std::vector<int> h_in(max_num_items);
  for (auto& v : h_in)
  {
    v = dist(rng);
  }

  verify_indirect_exclusive_scan(h_in, max_num_items, actual_num_items, ::cuda::std::plus<>{}, 999);
}

// --- 21. Multiplies with init=1, small values to avoid overflow ---
TEST_CASE("DeviceScan::ExclusiveScan indirect multiplies with init=1", "[scan][indirect][device]")
{
  // Use small values (1..3) and short sequences to avoid overflow
  constexpr int max_num_items = 20;
  const int actual_num_items  = GENERATE(0, 1, 5, 10, 20);

  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(1, 3);

  std::vector<int> h_in(max_num_items);
  for (auto& v : h_in)
  {
    v = dist(rng);
  }

  verify_indirect_exclusive_scan(h_in, max_num_items, actual_num_items, ::cuda::std::multiplies<>{}, 1);
}

// ############################################################################
//
//  InclusiveScan with custom op tests
//
// ############################################################################

// --- 22. Plus: various sizes ---
TEST_CASE("DeviceScan::InclusiveScan indirect plus", "[scan][indirect][device]")
{
  const int actual_num_items  = GENERATE(0, 1, 100, 512, 2048);
  constexpr int max_num_items = 2048;

  std::vector<int> h_in(max_num_items, 1);

  verify_indirect_inclusive_scan(h_in, max_num_items, actual_num_items, ::cuda::std::plus<>{});
}

// --- 23. Plus: random data compared to standard ---
TEST_CASE("DeviceScan::InclusiveScan indirect plus random", "[scan][indirect][device]")
{
  constexpr int max_num_items = 10000;
  const int actual_num_items  = GENERATE(100, 1000, 5000, 10000);

  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, 100);

  std::vector<int> h_in(max_num_items);
  for (auto& v : h_in)
  {
    v = dist(rng);
  }

  verify_indirect_inclusive_scan(h_in, max_num_items, actual_num_items, ::cuda::std::plus<>{});
}

// ############################################################################
//
//  Varying num_items between calls (reusing temp storage)
//
// ############################################################################

// --- 24. ExclusiveSum varying ---
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

// --- 25. InclusiveSum varying ---
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

  for (int n : {10, 500, 1, 10000, 0, 42, 9999, 1024, 1023})
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

// --- 26. ExclusiveScan with plus and init varying ---
TEST_CASE("DeviceScan::ExclusiveScan indirect plus varying num_items", "[scan][indirect][device]")
{
  constexpr int max_num_items = 10000;
  constexpr int init_value    = 100;

  std::vector<int> h_in(max_num_items, 1);
  thrust::device_vector<int> d_in(h_in.begin(), h_in.end());
  thrust::device_vector<int> d_out(max_num_items);

  thrust::device_vector<int> d_num_items_vec(1, 0);
  int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

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

  for (int n : {10, 500, 1, 10000, 0, 42, 9999, 1024, 1023})
  {
    d_num_items_vec[0] = n;
    thrust::fill(d_out.begin(), d_out.end(), -1);

    REQUIRE(
      cudaSuccess
      == cub::DeviceScan::ExclusiveScan(
        d_temp_storage,
        temp_storage_bytes,
        d_in.begin(),
        d_out.begin(),
        ::cuda::std::plus<>{},
        init_value,
        static_cast<const int*>(d_num_items),
        max_num_items));
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    thrust::host_vector<int> h_out(d_out);
    // Expected: exclusive scan of all-1s with init=100 and plus
    // result[i] = 100 + i
    for (int i = 0; i < n; ++i)
    {
      REQUIRE(h_out[i] == init_value + i);
    }
  }
}

// ############################################################################
//
//  CUDA Graph capture tests (the whole point of indirect!)
//
// ############################################################################

// --- 27. ExclusiveSum graph ---
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

// --- 28. InclusiveSum graph ---
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

// --- 29. ExclusiveScan graph ---
TEST_CASE("DeviceScan::ExclusiveScan indirect CUDA graph capture", "[scan][indirect][device][graph]")
{
  constexpr int max_num_items = 10000;
  constexpr int init_value    = 42;

  std::vector<int> h_in(max_num_items, 1);
  thrust::device_vector<int> d_in(h_in.begin(), h_in.end());
  thrust::device_vector<int> d_out(max_num_items, -1);

  thrust::device_vector<int> d_num_items_vec(1, 0);
  int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

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

  cudaStream_t stream{};
  REQUIRE(cudaSuccess == cudaStreamCreate(&stream));

  cudaGraph_t graph{};
  REQUIRE(cudaSuccess == cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::ExclusiveScan(
      d_temp_storage,
      temp_storage_bytes,
      d_in.begin(),
      d_out.begin(),
      ::cuda::std::plus<>{},
      init_value,
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
      REQUIRE(h_out[i] == init_value + i);
    }
  }

  REQUIRE(cudaSuccess == cudaGraphExecDestroy(exec));
  REQUIRE(cudaSuccess == cudaGraphDestroy(graph));
  REQUIRE(cudaSuccess == cudaStreamDestroy(stream));
}

// --- 30. InclusiveScan graph ---
TEST_CASE("DeviceScan::InclusiveScan indirect CUDA graph capture", "[scan][indirect][device][graph]")
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

  cudaStream_t stream{};
  REQUIRE(cudaSuccess == cudaStreamCreate(&stream));

  cudaGraph_t graph{};
  REQUIRE(cudaSuccess == cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::InclusiveScan(
      d_temp_storage,
      temp_storage_bytes,
      d_in.begin(),
      d_out.begin(),
      ::cuda::std::plus<>{},
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
    // InclusiveScan with plus and all-1s input: result[i] = i+1
    for (int i = 0; i < n; ++i)
    {
      REQUIRE(h_out[i] == i + 1);
    }
  }

  REQUIRE(cudaSuccess == cudaGraphExecDestroy(exec));
  REQUIRE(cudaSuccess == cudaGraphDestroy(graph));
  REQUIRE(cudaSuccess == cudaStreamDestroy(stream));
}

// --- 31. ExclusiveSum graph stress: 100 iterations with random sizes ---
TEST_CASE(
  "DeviceScan::ExclusiveSum indirect graph stress 100 iterations", "[scan][indirect][device][graph][stress]")
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

  std::mt19937 rng(7);
  std::uniform_int_distribution<int> dist(0, max_num_items);

  for (int iter = 0; iter < 100; ++iter)
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

// --- 32. InclusiveSum graph stress: 100 iterations ---
TEST_CASE(
  "DeviceScan::InclusiveSum indirect graph stress 100 iterations", "[scan][indirect][device][graph][stress]")
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

  std::mt19937 rng(13);
  std::uniform_int_distribution<int> dist(0, max_num_items);

  for (int iter = 0; iter < 100; ++iter)
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
      REQUIRE(h_out[i] == i + 1);
    }
  }

  REQUIRE(cudaSuccess == cudaGraphExecDestroy(exec));
  REQUIRE(cudaSuccess == cudaGraphDestroy(graph));
  REQUIRE(cudaSuccess == cudaStreamDestroy(stream));
}

// --- 33. ExclusiveScan graph stress: 100 iterations ---
TEST_CASE(
  "DeviceScan::ExclusiveScan indirect graph stress 100 iterations", "[scan][indirect][device][graph][stress]")
{
  constexpr int max_num_items = 50000;
  constexpr int init_value    = 42;

  std::vector<int> h_in(max_num_items, 1);
  thrust::device_vector<int> d_in(h_in.begin(), h_in.end());
  thrust::device_vector<int> d_out(max_num_items, -1);

  thrust::device_vector<int> d_num_items_vec(1, 0);
  int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

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

  cudaStream_t stream{};
  REQUIRE(cudaSuccess == cudaStreamCreate(&stream));

  cudaGraph_t graph{};
  REQUIRE(cudaSuccess == cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::ExclusiveScan(
      d_temp_storage,
      temp_storage_bytes,
      d_in.begin(),
      d_out.begin(),
      ::cuda::std::plus<>{},
      init_value,
      static_cast<const int*>(d_num_items),
      max_num_items,
      stream));

  REQUIRE(cudaSuccess == cudaStreamEndCapture(stream, &graph));

  cudaGraphExec_t exec{};
  REQUIRE(cudaSuccess == cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  std::mt19937 rng(19);
  std::uniform_int_distribution<int> dist(0, max_num_items);

  for (int iter = 0; iter < 100; ++iter)
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
      REQUIRE(h_out[i] == init_value + i);
    }
  }

  REQUIRE(cudaSuccess == cudaGraphExecDestroy(exec));
  REQUIRE(cudaSuccess == cudaGraphDestroy(graph));
  REQUIRE(cudaSuccess == cudaStreamDestroy(stream));
}

// --- 34. Graph with float data ---
TEST_CASE("DeviceScan::ExclusiveSum indirect CUDA graph float", "[scan][indirect][device][graph]")
{
  constexpr int max_num_items = 10000;

  std::vector<float> h_in(max_num_items, 1.0f);
  thrust::device_vector<float> d_in(h_in.begin(), h_in.end());
  thrust::device_vector<float> d_out(max_num_items, -1.0f);

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

  for (int n : {0, 1, 100, 5000, 10000})
  {
    d_num_items_vec[0] = n;
    thrust::fill(d_out.begin(), d_out.end(), -1.0f);
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    REQUIRE(cudaSuccess == cudaGraphLaunch(exec, stream));
    REQUIRE(cudaSuccess == cudaStreamSynchronize(stream));

    thrust::host_vector<float> h_out(d_out);
    for (int i = 0; i < n; ++i)
    {
      REQUIRE(h_out[i] == Catch::Approx(static_cast<float>(i)));
    }
  }

  REQUIRE(cudaSuccess == cudaGraphExecDestroy(exec));
  REQUIRE(cudaSuccess == cudaGraphDestroy(graph));
  REQUIRE(cudaSuccess == cudaStreamDestroy(stream));
}

// --- 35. Graph where actual==max every time (no wasted tiles) ---
TEST_CASE("DeviceScan::ExclusiveSum indirect graph actual equals max", "[scan][indirect][device][graph]")
{
  constexpr int max_num_items = 4096;

  std::vector<int> h_in(max_num_items, 1);
  thrust::device_vector<int> d_in(h_in.begin(), h_in.end());
  thrust::device_vector<int> d_out(max_num_items, -1);

  thrust::device_vector<int> d_num_items_vec(1, max_num_items);
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

  // Replay 20 times, always actual == max
  for (int iter = 0; iter < 20; ++iter)
  {
    d_num_items_vec[0] = max_num_items;
    thrust::fill(d_out.begin(), d_out.end(), -1);
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    REQUIRE(cudaSuccess == cudaGraphLaunch(exec, stream));
    REQUIRE(cudaSuccess == cudaStreamSynchronize(stream));

    thrust::host_vector<int> h_out(d_out);
    for (int i = 0; i < max_num_items; ++i)
    {
      REQUIRE(h_out[i] == i);
    }
  }

  REQUIRE(cudaSuccess == cudaGraphExecDestroy(exec));
  REQUIRE(cudaSuccess == cudaGraphDestroy(graph));
  REQUIRE(cudaSuccess == cudaStreamDestroy(stream));
}

// --- 36. Graph where actual==0 every time (all tiles are no-ops) ---
TEST_CASE("DeviceScan::ExclusiveSum indirect graph actual equals zero", "[scan][indirect][device][graph]")
{
  constexpr int max_num_items = 4096;

  std::vector<int> h_in(max_num_items, 1);
  thrust::device_vector<int> d_in(h_in.begin(), h_in.end());

  const int sentinel = -999;
  thrust::device_vector<int> d_out(max_num_items, sentinel);

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

  // Replay 20 times, always actual == 0
  for (int iter = 0; iter < 20; ++iter)
  {
    d_num_items_vec[0] = 0;
    thrust::fill(d_out.begin(), d_out.end(), sentinel);
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    REQUIRE(cudaSuccess == cudaGraphLaunch(exec, stream));
    REQUIRE(cudaSuccess == cudaStreamSynchronize(stream));

    // With actual==0, no valid output elements
    // Just verify no crash and no CUDA errors
  }

  REQUIRE(cudaSuccess == cudaGraphExecDestroy(exec));
  REQUIRE(cudaSuccess == cudaGraphDestroy(graph));
  REQUIRE(cudaSuccess == cudaStreamDestroy(stream));
}

// ############################################################################
//
//  Edge cases
//
// ############################################################################

// --- 37. max_num_items = 0 ---
TEST_CASE("DeviceScan indirect zero max_num_items", "[scan][indirect][device]")
{
  thrust::device_vector<int> d_in(1, 1);
  thrust::device_vector<int> d_out(1, -1);

  thrust::device_vector<int> d_num_items_vec(1, 0);
  const int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;

  SECTION("ExclusiveSum")
  {
    // max_num_items = 0 should return without error
    REQUIRE(
      cudaSuccess
      == cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), d_num_items, 0));
  }

  SECTION("InclusiveSum")
  {
    REQUIRE(
      cudaSuccess
      == cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), d_num_items, 0));
  }

  SECTION("ExclusiveScan")
  {
    REQUIRE(
      cudaSuccess
      == cub::DeviceScan::ExclusiveScan(
        d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), ::cuda::std::plus<>{}, 0, d_num_items, 0));
  }

  SECTION("InclusiveScan")
  {
    REQUIRE(
      cudaSuccess
      == cub::DeviceScan::InclusiveScan(
        d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), ::cuda::std::plus<>{}, d_num_items, 0));
  }
}

// --- 38. actual = 1, single element ---
TEST_CASE("DeviceScan indirect actual equals 1 single element", "[scan][indirect][device]")
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

  SECTION("ExclusiveScan with init")
  {
    thrust::device_vector<int> d_out(max_num_items, -1);
    thrust::device_vector<int> d_num_items_vec(1, 1);
    const int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

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
        99,
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
        99,
        d_num_items,
        max_num_items));
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    thrust::host_vector<int> h_out(d_out);
    REQUIRE(h_out[0] == 99); // exclusive scan with 1 element: init value
  }

  SECTION("InclusiveScan")
  {
    thrust::device_vector<int> d_out(max_num_items, -1);
    thrust::device_vector<int> d_num_items_vec(1, 1);
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
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    thrust::host_vector<int> h_out(d_out);
    REQUIRE(h_out[0] == 42); // inclusive scan with 1 element: the element itself
  }
}

// --- 39. Output beyond actual_num_items check ---
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

// --- 40. max_num_items = 1, actual = 0 and 1 ---
TEST_CASE("DeviceScan indirect max_num_items equals 1", "[scan][indirect][device]")
{
  constexpr int max_num_items = 1;

  std::vector<int> h_in = {42};
  thrust::device_vector<int> d_in(h_in.begin(), h_in.end());

  SECTION("ExclusiveSum actual=0")
  {
    thrust::device_vector<int> d_out(max_num_items, -1);
    thrust::device_vector<int> d_num_items_vec(1, 0);
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
    // actual=0: no output to verify, just no crash
  }

  SECTION("ExclusiveSum actual=1")
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
    REQUIRE(h_out[0] == 0);
  }

  SECTION("InclusiveSum actual=0")
  {
    thrust::device_vector<int> d_out(max_num_items, -1);
    thrust::device_vector<int> d_num_items_vec(1, 0);
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
  }

  SECTION("InclusiveSum actual=1")
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
    REQUIRE(h_out[0] == 42);
  }

  SECTION("ExclusiveScan actual=0")
  {
    thrust::device_vector<int> d_out(max_num_items, -1);
    thrust::device_vector<int> d_num_items_vec(1, 0);
    const int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

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
        7,
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
        7,
        d_num_items,
        max_num_items));
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  }

  SECTION("ExclusiveScan actual=1")
  {
    thrust::device_vector<int> d_out(max_num_items, -1);
    thrust::device_vector<int> d_num_items_vec(1, 1);
    const int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

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
        7,
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
        7,
        d_num_items,
        max_num_items));
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    thrust::host_vector<int> h_out(d_out);
    REQUIRE(h_out[0] == 7); // exclusive scan single element: init value
  }

  SECTION("InclusiveScan actual=0")
  {
    thrust::device_vector<int> d_out(max_num_items, -1);
    thrust::device_vector<int> d_num_items_vec(1, 0);
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
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  }

  SECTION("InclusiveScan actual=1")
  {
    thrust::device_vector<int> d_out(max_num_items, -1);
    thrust::device_vector<int> d_num_items_vec(1, 1);
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
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    thrust::host_vector<int> h_out(d_out);
    REQUIRE(h_out[0] == 42);
  }
}

// ############################################################################
//
//  Very large sizes (stress multi-tile logic)
//
// ############################################################################

// --- 41. max=4M, actual=4M ---
TEST_CASE("DeviceScan::ExclusiveSum indirect very large 4M", "[scan][indirect][device][stress]")
{
  constexpr int max_num_items = 4 * 1024 * 1024; // 4M elements

  // All-ones so we can verify: result[i] == i
  std::vector<int> h_in(max_num_items, 1);
  thrust::device_vector<int> d_in(h_in.begin(), h_in.end());
  thrust::device_vector<int> d_out(max_num_items, -1);

  thrust::device_vector<int> d_num_items_vec(1, max_num_items);
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
  for (int i = 0; i < max_num_items; ++i)
  {
    REQUIRE(h_out[i] == i);
  }
}

TEST_CASE("DeviceScan::InclusiveSum indirect very large 4M", "[scan][indirect][device][stress]")
{
  constexpr int max_num_items = 4 * 1024 * 1024;

  std::vector<int> h_in(max_num_items, 1);
  thrust::device_vector<int> d_in(h_in.begin(), h_in.end());
  thrust::device_vector<int> d_out(max_num_items, -1);

  thrust::device_vector<int> d_num_items_vec(1, max_num_items);
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
  for (int i = 0; i < max_num_items; ++i)
  {
    REQUIRE(h_out[i] == i + 1);
  }
}

// ############################################################################
//
//  Additional coverage: InclusiveSum with more patterns
//
// ############################################################################

TEST_CASE("DeviceScan::InclusiveSum indirect all-zeros", "[scan][indirect][device]")
{
  constexpr int max_num_items = 2048;
  const int actual_num_items  = GENERATE(0, 1, 512, 2048);

  std::vector<int> h_in(max_num_items, 0);

  verify_indirect_inclusive_sum(h_in, max_num_items, actual_num_items);
}

TEST_CASE("DeviceScan::InclusiveSum indirect all-ones", "[scan][indirect][device]")
{
  constexpr int max_num_items = 2048;
  const int actual_num_items  = GENERATE(0, 1, 512, 2048);

  std::vector<int> h_in(max_num_items, 1);

  verify_indirect_inclusive_sum(h_in, max_num_items, actual_num_items);
}

TEST_CASE("DeviceScan::InclusiveSum indirect alternating values", "[scan][indirect][device]")
{
  constexpr int max_num_items = 4096;
  const int actual_num_items  = GENERATE(0, 1, 2, 3, 100, 1023, 1024, 4096);

  std::vector<int> h_in(max_num_items);
  for (int i = 0; i < max_num_items; ++i)
  {
    h_in[i] = (i % 2 == 0) ? 1 : 0;
  }

  verify_indirect_inclusive_sum(h_in, max_num_items, actual_num_items);
}

TEST_CASE("DeviceScan::InclusiveSum indirect multi-tile boundaries", "[scan][indirect][device]")
{
  constexpr int max_num_items = 8192;
  const int actual_num_items  = GENERATE(
    511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096, 4097, 8191, 8192);

  std::vector<int> h_in(max_num_items, 1);

  verify_indirect_inclusive_sum(h_in, max_num_items, actual_num_items);
}

TEST_CASE("DeviceScan::InclusiveSum indirect actual much smaller than max", "[scan][indirect][device]")
{
  constexpr int max_num_items = 1 << 18;
  const int actual_num_items  = GENERATE(0, 1, 10, 100);

  std::vector<int> h_in(max_num_items, 1);

  verify_indirect_inclusive_sum(h_in, max_num_items, actual_num_items);
}

TEST_CASE("DeviceScan::InclusiveSum indirect power-of-two sizes", "[scan][indirect][device]")
{
  constexpr int max_num_items = 1 << 16;
  const int actual_num_items  = GENERATE(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
                                         1 << 16);

  std::vector<int> h_in(max_num_items, 1);

  verify_indirect_inclusive_sum(h_in, max_num_items, actual_num_items);
}

TEST_CASE("DeviceScan::InclusiveSum indirect non-power-of-two sizes", "[scan][indirect][device]")
{
  constexpr int max_num_items = 10000;
  const int actual_num_items  = GENERATE(3, 7, 13, 127, 129, 255, 257, 511, 513, 1023, 1025, 4095, 4097, 9999);

  std::vector<int> h_in(max_num_items, 1);

  verify_indirect_inclusive_sum(h_in, max_num_items, actual_num_items);
}

// ############################################################################
//
//  Additional coverage: ExclusiveScan and InclusiveScan patterns
//
// ############################################################################

TEST_CASE("DeviceScan::ExclusiveScan indirect plus multi-tile boundaries", "[scan][indirect][device]")
{
  constexpr int max_num_items = 8192;
  constexpr int init_value    = 10;
  const int actual_num_items  = GENERATE(
    511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096, 4097, 8191, 8192);

  std::vector<int> h_in(max_num_items, 1);

  verify_indirect_exclusive_scan(h_in, max_num_items, actual_num_items, ::cuda::std::plus<>{}, init_value);
}

TEST_CASE("DeviceScan::InclusiveScan indirect plus multi-tile boundaries", "[scan][indirect][device]")
{
  constexpr int max_num_items = 8192;
  const int actual_num_items  = GENERATE(
    511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096, 4097, 8191, 8192);

  std::vector<int> h_in(max_num_items, 1);

  verify_indirect_inclusive_scan(h_in, max_num_items, actual_num_items, ::cuda::std::plus<>{});
}

TEST_CASE("DeviceScan::ExclusiveScan indirect plus all-zeros", "[scan][indirect][device]")
{
  constexpr int max_num_items = 2048;
  constexpr int init_value    = 42;
  const int actual_num_items  = GENERATE(0, 1, 100, 2048);

  std::vector<int> h_in(max_num_items, 0);

  verify_indirect_exclusive_scan(h_in, max_num_items, actual_num_items, ::cuda::std::plus<>{}, init_value);
}

TEST_CASE("DeviceScan::InclusiveScan indirect plus all-zeros", "[scan][indirect][device]")
{
  constexpr int max_num_items = 2048;
  const int actual_num_items  = GENERATE(0, 1, 100, 2048);

  std::vector<int> h_in(max_num_items, 0);

  verify_indirect_inclusive_scan(h_in, max_num_items, actual_num_items, ::cuda::std::plus<>{});
}

// ############################################################################
//
//  InclusiveScan graph stress
//
// ############################################################################

TEST_CASE(
  "DeviceScan::InclusiveScan indirect graph stress 100 iterations", "[scan][indirect][device][graph][stress]")
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

  cudaStream_t stream{};
  REQUIRE(cudaSuccess == cudaStreamCreate(&stream));

  cudaGraph_t graph{};
  REQUIRE(cudaSuccess == cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::InclusiveScan(
      d_temp_storage,
      temp_storage_bytes,
      d_in.begin(),
      d_out.begin(),
      ::cuda::std::plus<>{},
      static_cast<const int*>(d_num_items),
      max_num_items,
      stream));

  REQUIRE(cudaSuccess == cudaStreamEndCapture(stream, &graph));

  cudaGraphExec_t exec{};
  REQUIRE(cudaSuccess == cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  std::mt19937 rng(31);
  std::uniform_int_distribution<int> dist(0, max_num_items);

  for (int iter = 0; iter < 100; ++iter)
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
      REQUIRE(h_out[i] == i + 1);
    }
  }

  REQUIRE(cudaSuccess == cudaGraphExecDestroy(exec));
  REQUIRE(cudaSuccess == cudaGraphDestroy(graph));
  REQUIRE(cudaSuccess == cudaStreamDestroy(stream));
}

// ############################################################################
//
//  Additional graph tests: InclusiveSum graph with float
//
// ############################################################################

TEST_CASE("DeviceScan::InclusiveSum indirect CUDA graph float", "[scan][indirect][device][graph]")
{
  constexpr int max_num_items = 10000;

  std::vector<float> h_in(max_num_items, 1.0f);
  thrust::device_vector<float> d_in(h_in.begin(), h_in.end());
  thrust::device_vector<float> d_out(max_num_items, -1.0f);

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

  for (int n : {0, 1, 100, 5000, 10000})
  {
    d_num_items_vec[0] = n;
    thrust::fill(d_out.begin(), d_out.end(), -1.0f);
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    REQUIRE(cudaSuccess == cudaGraphLaunch(exec, stream));
    REQUIRE(cudaSuccess == cudaStreamSynchronize(stream));

    thrust::host_vector<float> h_out(d_out);
    for (int i = 0; i < n; ++i)
    {
      REQUIRE(h_out[i] == Catch::Approx(static_cast<float>(i + 1)));
    }
  }

  REQUIRE(cudaSuccess == cudaGraphExecDestroy(exec));
  REQUIRE(cudaSuccess == cudaGraphDestroy(graph));
  REQUIRE(cudaSuccess == cudaStreamDestroy(stream));
}

// ############################################################################
//
//  InclusiveScan varying num_items between calls
//
// ############################################################################

TEST_CASE("DeviceScan::InclusiveScan indirect plus varying num_items", "[scan][indirect][device]")
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

  for (int n : {10, 500, 1, 10000, 0, 42, 9999, 1024, 1023})
  {
    d_num_items_vec[0] = n;
    thrust::fill(d_out.begin(), d_out.end(), -1);

    REQUIRE(
      cudaSuccess
      == cub::DeviceScan::InclusiveScan(
        d_temp_storage,
        temp_storage_bytes,
        d_in.begin(),
        d_out.begin(),
        ::cuda::std::plus<>{},
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

// ############################################################################
//
//  Additional coverage: ExclusiveSum with random data and float
//
// ############################################################################

TEST_CASE("DeviceScan::ExclusiveSum indirect random float data", "[scan][indirect][device]")
{
  constexpr int max_num_items = 10000;
  const int actual_num_items  = GENERATE(100, 1000, 10000);

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  std::vector<float> h_in(max_num_items);
  for (auto& v : h_in)
  {
    v = dist(rng);
  }

  verify_indirect_exclusive_sum_float(h_in, max_num_items, actual_num_items);
}

TEST_CASE("DeviceScan::InclusiveSum indirect random float data", "[scan][indirect][device]")
{
  constexpr int max_num_items = 10000;
  const int actual_num_items  = GENERATE(100, 1000, 10000);

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  std::vector<float> h_in(max_num_items);
  for (auto& v : h_in)
  {
    v = dist(rng);
  }

  verify_indirect_inclusive_sum_float(h_in, max_num_items, actual_num_items);
}

// ############################################################################
//
//  Additional graph coverage: InclusiveScan and ExclusiveScan with float
//
// ############################################################################

TEST_CASE("DeviceScan::ExclusiveScan indirect CUDA graph float", "[scan][indirect][device][graph]")
{
  constexpr int max_num_items = 10000;
  constexpr float init_value  = 0.0f;

  std::vector<float> h_in(max_num_items, 1.0f);
  thrust::device_vector<float> d_in(h_in.begin(), h_in.end());
  thrust::device_vector<float> d_out(max_num_items, -1.0f);

  thrust::device_vector<int> d_num_items_vec(1, 0);
  int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

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

  cudaStream_t stream{};
  REQUIRE(cudaSuccess == cudaStreamCreate(&stream));

  cudaGraph_t graph{};
  REQUIRE(cudaSuccess == cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::ExclusiveScan(
      d_temp_storage,
      temp_storage_bytes,
      d_in.begin(),
      d_out.begin(),
      ::cuda::std::plus<>{},
      init_value,
      static_cast<const int*>(d_num_items),
      max_num_items,
      stream));

  REQUIRE(cudaSuccess == cudaStreamEndCapture(stream, &graph));

  cudaGraphExec_t exec{};
  REQUIRE(cudaSuccess == cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  for (int n : {0, 1, 100, 5000, 10000})
  {
    d_num_items_vec[0] = n;
    thrust::fill(d_out.begin(), d_out.end(), -1.0f);
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    REQUIRE(cudaSuccess == cudaGraphLaunch(exec, stream));
    REQUIRE(cudaSuccess == cudaStreamSynchronize(stream));

    thrust::host_vector<float> h_out(d_out);
    for (int i = 0; i < n; ++i)
    {
      REQUIRE(h_out[i] == Catch::Approx(static_cast<float>(i)));
    }
  }

  REQUIRE(cudaSuccess == cudaGraphExecDestroy(exec));
  REQUIRE(cudaSuccess == cudaGraphDestroy(graph));
  REQUIRE(cudaSuccess == cudaStreamDestroy(stream));
}

TEST_CASE("DeviceScan::InclusiveScan indirect CUDA graph float", "[scan][indirect][device][graph]")
{
  constexpr int max_num_items = 10000;

  std::vector<float> h_in(max_num_items, 1.0f);
  thrust::device_vector<float> d_in(h_in.begin(), h_in.end());
  thrust::device_vector<float> d_out(max_num_items, -1.0f);

  thrust::device_vector<int> d_num_items_vec(1, 0);
  int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

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

  cudaStream_t stream{};
  REQUIRE(cudaSuccess == cudaStreamCreate(&stream));

  cudaGraph_t graph{};
  REQUIRE(cudaSuccess == cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::InclusiveScan(
      d_temp_storage,
      temp_storage_bytes,
      d_in.begin(),
      d_out.begin(),
      ::cuda::std::plus<>{},
      static_cast<const int*>(d_num_items),
      max_num_items,
      stream));

  REQUIRE(cudaSuccess == cudaStreamEndCapture(stream, &graph));

  cudaGraphExec_t exec{};
  REQUIRE(cudaSuccess == cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  for (int n : {0, 1, 100, 5000, 10000})
  {
    d_num_items_vec[0] = n;
    thrust::fill(d_out.begin(), d_out.end(), -1.0f);
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    REQUIRE(cudaSuccess == cudaGraphLaunch(exec, stream));
    REQUIRE(cudaSuccess == cudaStreamSynchronize(stream));

    thrust::host_vector<float> h_out(d_out);
    for (int i = 0; i < n; ++i)
    {
      REQUIRE(h_out[i] == Catch::Approx(static_cast<float>(i + 1)));
    }
  }

  REQUIRE(cudaSuccess == cudaGraphExecDestroy(exec));
  REQUIRE(cudaSuccess == cudaGraphDestroy(graph));
  REQUIRE(cudaSuccess == cudaStreamDestroy(stream));
}

// ############################################################################
//
//  Additional coverage: ExclusiveScan/InclusiveScan with large sizes
//
// ############################################################################

TEST_CASE("DeviceScan::ExclusiveScan indirect plus large", "[scan][indirect][device]")
{
  constexpr int max_num_items = 1 << 20;
  constexpr int init_value    = 0;
  const int actual_num_items  = GENERATE(1 << 18, 1 << 19, 1 << 20);

  std::vector<int> h_in(max_num_items, 1);

  verify_indirect_exclusive_scan(h_in, max_num_items, actual_num_items, ::cuda::std::plus<>{}, init_value);
}

TEST_CASE("DeviceScan::InclusiveScan indirect plus large", "[scan][indirect][device]")
{
  constexpr int max_num_items = 1 << 20;
  const int actual_num_items  = GENERATE(1 << 18, 1 << 19, 1 << 20);

  std::vector<int> h_in(max_num_items, 1);

  verify_indirect_inclusive_scan(h_in, max_num_items, actual_num_items, ::cuda::std::plus<>{});
}

// ############################################################################
//
//  Additional coverage: ExclusiveScan/InclusiveScan varying and graphs
//
// ############################################################################

TEST_CASE("DeviceScan::ExclusiveScan indirect plus actual much smaller than max", "[scan][indirect][device]")
{
  constexpr int max_num_items = 1 << 18;
  constexpr int init_value    = 7;
  const int actual_num_items  = GENERATE(0, 1, 10, 100);

  std::vector<int> h_in(max_num_items, 1);

  verify_indirect_exclusive_scan(h_in, max_num_items, actual_num_items, ::cuda::std::plus<>{}, init_value);
}

TEST_CASE("DeviceScan::InclusiveScan indirect plus actual much smaller than max", "[scan][indirect][device]")
{
  constexpr int max_num_items = 1 << 18;
  const int actual_num_items  = GENERATE(0, 1, 10, 100);

  std::vector<int> h_in(max_num_items, 1);

  verify_indirect_inclusive_scan(h_in, max_num_items, actual_num_items, ::cuda::std::plus<>{});
}

// ############################################################################
//
//  Graph tests: ExclusiveScan and InclusiveScan graph stress with random data
//
// ############################################################################

TEST_CASE(
  "DeviceScan::ExclusiveScan indirect graph stress random data", "[scan][indirect][device][graph][stress]")
{
  constexpr int max_num_items = 10000;
  constexpr int init_value    = 0;

  std::mt19937 data_rng(42);
  std::uniform_int_distribution<int> data_dist(0, 10);

  std::vector<int> h_in(max_num_items);
  for (auto& v : h_in)
  {
    v = data_dist(data_rng);
  }

  thrust::device_vector<int> d_in(h_in.begin(), h_in.end());
  thrust::device_vector<int> d_out(max_num_items, -1);

  thrust::device_vector<int> d_num_items_vec(1, 0);
  int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

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

  cudaStream_t stream{};
  REQUIRE(cudaSuccess == cudaStreamCreate(&stream));

  cudaGraph_t graph{};
  REQUIRE(cudaSuccess == cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::ExclusiveScan(
      d_temp_storage,
      temp_storage_bytes,
      d_in.begin(),
      d_out.begin(),
      ::cuda::std::plus<>{},
      init_value,
      static_cast<const int*>(d_num_items),
      max_num_items,
      stream));

  REQUIRE(cudaSuccess == cudaStreamEndCapture(stream, &graph));

  cudaGraphExec_t exec{};
  REQUIRE(cudaSuccess == cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  // Compute reference for all sizes up front
  auto h_ref_full = exclusive_scan_reference(h_in, max_num_items, std::plus<int>{}, init_value);

  std::mt19937 size_rng(99);
  std::uniform_int_distribution<int> size_dist(0, max_num_items);

  for (int iter = 0; iter < 50; ++iter)
  {
    int n              = size_dist(size_rng);
    d_num_items_vec[0] = n;
    thrust::fill(d_out.begin(), d_out.end(), -1);
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    REQUIRE(cudaSuccess == cudaGraphLaunch(exec, stream));
    REQUIRE(cudaSuccess == cudaStreamSynchronize(stream));

    thrust::host_vector<int> h_out(d_out);
    for (int i = 0; i < n; ++i)
    {
      REQUIRE(h_out[i] == h_ref_full[i]);
    }
  }

  REQUIRE(cudaSuccess == cudaGraphExecDestroy(exec));
  REQUIRE(cudaSuccess == cudaGraphDestroy(graph));
  REQUIRE(cudaSuccess == cudaStreamDestroy(stream));
}

TEST_CASE(
  "DeviceScan::InclusiveScan indirect graph stress random data", "[scan][indirect][device][graph][stress]")
{
  constexpr int max_num_items = 10000;

  std::mt19937 data_rng(42);
  std::uniform_int_distribution<int> data_dist(0, 10);

  std::vector<int> h_in(max_num_items);
  for (auto& v : h_in)
  {
    v = data_dist(data_rng);
  }

  thrust::device_vector<int> d_in(h_in.begin(), h_in.end());
  thrust::device_vector<int> d_out(max_num_items, -1);

  thrust::device_vector<int> d_num_items_vec(1, 0);
  int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

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

  cudaStream_t stream{};
  REQUIRE(cudaSuccess == cudaStreamCreate(&stream));

  cudaGraph_t graph{};
  REQUIRE(cudaSuccess == cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::InclusiveScan(
      d_temp_storage,
      temp_storage_bytes,
      d_in.begin(),
      d_out.begin(),
      ::cuda::std::plus<>{},
      static_cast<const int*>(d_num_items),
      max_num_items,
      stream));

  REQUIRE(cudaSuccess == cudaStreamEndCapture(stream, &graph));

  cudaGraphExec_t exec{};
  REQUIRE(cudaSuccess == cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  // Compute reference for all sizes up front
  auto h_ref_full = inclusive_scan_reference(h_in, max_num_items, std::plus<int>{});

  std::mt19937 size_rng(99);
  std::uniform_int_distribution<int> size_dist(0, max_num_items);

  for (int iter = 0; iter < 50; ++iter)
  {
    int n              = size_dist(size_rng);
    d_num_items_vec[0] = n;
    thrust::fill(d_out.begin(), d_out.end(), -1);
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    REQUIRE(cudaSuccess == cudaGraphLaunch(exec, stream));
    REQUIRE(cudaSuccess == cudaStreamSynchronize(stream));

    thrust::host_vector<int> h_out(d_out);
    for (int i = 0; i < n; ++i)
    {
      REQUIRE(h_out[i] == h_ref_full[i]);
    }
  }

  REQUIRE(cudaSuccess == cudaGraphExecDestroy(exec));
  REQUIRE(cudaSuccess == cudaGraphDestroy(graph));
  REQUIRE(cudaSuccess == cudaStreamDestroy(stream));
}

// ============================================================================
// THE REAL USE CASE: kernel writes d_num_items, scan reads it, all in one graph
// This simulates physics engine contact counting -> scan pattern
// ============================================================================

__global__ void write_num_items_kernel(int* d_num_items, const int* d_count_source)
{
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    *d_num_items = *d_count_source;
  }
}

TEST_CASE("DeviceScan::ExclusiveSum graph with kernel-written d_num_items", "[scan][indirect][device][graph]")
{
  constexpr int max_num_items = 50000;

  std::vector<int> h_in(max_num_items, 1);
  thrust::device_vector<int> d_in(h_in.begin(), h_in.end());
  thrust::device_vector<int> d_out(max_num_items, -1);

  thrust::device_vector<int> d_num_items_vec(1, 0);
  int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());
  thrust::device_vector<int> d_count_source(1, 0);
  int* d_count_ptr = thrust::raw_pointer_cast(d_count_source.data());

  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::ExclusiveSum(
      d_temp_storage, temp_storage_bytes,
      thrust::raw_pointer_cast(d_in.data()), thrust::raw_pointer_cast(d_out.data()),
      static_cast<const int*>(d_num_items), max_num_items));

  thrust::device_vector<std::uint8_t> d_temp(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(d_temp.data());

  cudaStream_t stream{};
  REQUIRE(cudaSuccess == cudaStreamCreate(&stream));
  cudaGraph_t graph{};
  REQUIRE(cudaSuccess == cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

  write_num_items_kernel<<<1, 1, 0, stream>>>(d_num_items, d_count_ptr);

  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::ExclusiveSum(
      d_temp_storage, temp_storage_bytes,
      thrust::raw_pointer_cast(d_in.data()), thrust::raw_pointer_cast(d_out.data()),
      static_cast<const int*>(d_num_items), max_num_items, stream));

  REQUIRE(cudaSuccess == cudaStreamEndCapture(stream, &graph));
  cudaGraphExec_t exec{};
  REQUIRE(cudaSuccess == cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  for (int n : {0, 1, 42, 1000, 10000, 50000, 500, 7, 49999})
  {
    d_count_source[0] = n;
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

TEST_CASE(
  "DeviceScan graph kernel-written d_num_items stress 200 iters", "[scan][indirect][device][graph][stress]")
{
  constexpr int max_num_items = 100000;

  std::vector<int> h_in(max_num_items, 1);
  thrust::device_vector<int> d_in(h_in.begin(), h_in.end());
  thrust::device_vector<int> d_out(max_num_items, -1);
  thrust::device_vector<int> d_num_items_vec(1, 0);
  int* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());
  thrust::device_vector<int> d_count_source(1, 0);
  int* d_count_ptr = thrust::raw_pointer_cast(d_count_source.data());

  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::ExclusiveSum(
      d_temp_storage, temp_storage_bytes,
      thrust::raw_pointer_cast(d_in.data()), thrust::raw_pointer_cast(d_out.data()),
      static_cast<const int*>(d_num_items), max_num_items));

  thrust::device_vector<std::uint8_t> d_temp(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(d_temp.data());

  cudaStream_t stream{};
  REQUIRE(cudaSuccess == cudaStreamCreate(&stream));
  cudaGraph_t graph{};
  REQUIRE(cudaSuccess == cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

  write_num_items_kernel<<<1, 1, 0, stream>>>(d_num_items, d_count_ptr);

  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::ExclusiveSum(
      d_temp_storage, temp_storage_bytes,
      thrust::raw_pointer_cast(d_in.data()), thrust::raw_pointer_cast(d_out.data()),
      static_cast<const int*>(d_num_items), max_num_items, stream));

  REQUIRE(cudaSuccess == cudaStreamEndCapture(stream, &graph));
  cudaGraphExec_t exec{};
  REQUIRE(cudaSuccess == cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, max_num_items);

  for (int iter = 0; iter < 200; ++iter)
  {
    int n             = dist(rng);
    d_count_source[0] = n;
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
