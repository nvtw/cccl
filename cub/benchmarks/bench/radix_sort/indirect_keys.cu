// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

// Benchmark for indirect (device-accessible num_items) radix sort vs standard radix sort.
// Includes CUDA graph capture benchmarks.

#include <cub/device/device_radix_sort.cuh>

#include <thrust/device_vector.h>

#include <nvbench_helper.cuh>

// ============================================================================
// Stream-based benchmarks
// ============================================================================

template <typename T, typename OffsetT>
void indirect_radix_sort_keys(nvbench::state& state, nvbench::type_list<T, OffsetT>)
try
{
  using offset_t = cub::detail::choose_offset_t<OffsetT>;

  const auto elements     = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const double fill_ratio = state.get_float64("FillRatio");
  const bit_entropy entropy = str_to_entropy(state.get_string("Entropy"));
  const auto actual_items = static_cast<offset_t>(elements * fill_ratio);
  const auto max_items    = static_cast<offset_t>(elements);

  constexpr int begin_bit = 0;
  constexpr int end_bit   = sizeof(T) * 8;

  thrust::device_vector<T> input_1 = generate(elements, entropy);
  thrust::device_vector<T> input_2(elements);

  thrust::device_vector<offset_t> d_num_items_vec(1, actual_items);
  const offset_t* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

  state.add_element_count(actual_items);
  state.add_global_memory_reads<T>(actual_items, "Size");
  state.add_global_memory_writes<T>(actual_items);

  size_t tmp_size{};
  cub::DeviceRadixSort::SortKeys(
    nullptr,
    tmp_size,
    thrust::raw_pointer_cast(input_1.data()),
    thrust::raw_pointer_cast(input_2.data()),
    d_num_items,
    max_items,
    begin_bit,
    end_bit);

  thrust::device_vector<nvbench::uint8_t> tmp(tmp_size, thrust::no_init);

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    cub::DeviceRadixSort::SortKeys(
      thrust::raw_pointer_cast(tmp.data()),
      tmp_size,
      thrust::raw_pointer_cast(input_1.data()),
      thrust::raw_pointer_cast(input_2.data()),
      d_num_items,
      max_items,
      begin_bit,
      end_bit,
      launch.get_stream());
  });
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

template <typename T, typename OffsetT>
void standard_radix_sort_keys(nvbench::state& state, nvbench::type_list<T, OffsetT>)
try
{
  using offset_t = cub::detail::choose_offset_t<OffsetT>;

  const auto elements       = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const bit_entropy entropy = str_to_entropy(state.get_string("Entropy"));

  constexpr int begin_bit = 0;
  constexpr int end_bit   = sizeof(T) * 8;

  thrust::device_vector<T> buffer_1 = generate(elements, entropy);
  thrust::device_vector<T> buffer_2(elements);

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(elements);

  size_t tmp_size{};
  cub::DeviceRadixSort::SortKeys(
    nullptr,
    tmp_size,
    thrust::raw_pointer_cast(buffer_1.data()),
    thrust::raw_pointer_cast(buffer_2.data()),
    static_cast<offset_t>(elements),
    begin_bit,
    end_bit);

  thrust::device_vector<nvbench::uint8_t> tmp(tmp_size, thrust::no_init);

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    cub::DeviceRadixSort::SortKeys(
      thrust::raw_pointer_cast(tmp.data()),
      tmp_size,
      thrust::raw_pointer_cast(buffer_1.data()),
      thrust::raw_pointer_cast(buffer_2.data()),
      static_cast<offset_t>(elements),
      begin_bit,
      end_bit,
      launch.get_stream());
  });
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

// ============================================================================
// CUDA graph benchmarks
// ============================================================================

template <typename T, typename OffsetT>
void indirect_radix_sort_keys_graph(nvbench::state& state, nvbench::type_list<T, OffsetT>)
try
{
  using offset_t = cub::detail::choose_offset_t<OffsetT>;

  const auto elements     = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const double fill_ratio = state.get_float64("FillRatio");
  const bit_entropy entropy = str_to_entropy(state.get_string("Entropy"));
  const auto actual_items = static_cast<offset_t>(elements * fill_ratio);
  const auto max_items    = static_cast<offset_t>(elements);

  constexpr int begin_bit = 0;
  constexpr int end_bit   = sizeof(T) * 8;

  thrust::device_vector<T> input_1 = generate(elements, entropy);
  thrust::device_vector<T> input_2(elements);

  thrust::device_vector<offset_t> d_num_items_vec(1, actual_items);
  const offset_t* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

  state.add_element_count(actual_items);
  state.add_global_memory_reads<T>(actual_items, "Size");
  state.add_global_memory_writes<T>(actual_items);

  size_t tmp_size{};
  cub::DeviceRadixSort::SortKeys(
    nullptr,
    tmp_size,
    thrust::raw_pointer_cast(input_1.data()),
    thrust::raw_pointer_cast(input_2.data()),
    d_num_items,
    max_items,
    begin_bit,
    end_bit);

  thrust::device_vector<nvbench::uint8_t> tmp(tmp_size, thrust::no_init);

  // Capture graph
  cudaStream_t capture_stream{};
  cudaStreamCreate(&capture_stream);

  cudaGraph_t graph{};
  cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal);

  cub::DeviceRadixSort::SortKeys(
    thrust::raw_pointer_cast(tmp.data()),
    tmp_size,
    thrust::raw_pointer_cast(input_1.data()),
    thrust::raw_pointer_cast(input_2.data()),
    d_num_items,
    max_items,
    begin_bit,
    end_bit,
    capture_stream);

  cudaStreamEndCapture(capture_stream, &graph);

  cudaGraphExec_t graph_exec{};
  cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    cudaGraphLaunch(graph_exec, launch.get_stream());
  });

  cudaGraphExecDestroy(graph_exec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(capture_stream);
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

using sort_types = nvbench::type_list<nvbench::int32_t, nvbench::uint32_t>;

NVBENCH_BENCH_TYPES(indirect_radix_sort_keys, NVBENCH_TYPE_AXES(sort_types, offset_types))
  .set_name("indirect_radix_sort_keys")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_float64_axis("FillRatio", {0.01, 0.1, 0.5, 1.0})
  .add_string_axis("Entropy", {"1.000", "0.201"});

NVBENCH_BENCH_TYPES(standard_radix_sort_keys, NVBENCH_TYPE_AXES(sort_types, offset_types))
  .set_name("standard_radix_sort_keys")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_string_axis("Entropy", {"1.000", "0.201"});

NVBENCH_BENCH_TYPES(indirect_radix_sort_keys_graph, NVBENCH_TYPE_AXES(sort_types, offset_types))
  .set_name("indirect_radix_sort_keys_graph")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_float64_axis("FillRatio", {0.01, 0.1, 0.5, 1.0})
  .add_string_axis("Entropy", {"1.000", "0.201"});
