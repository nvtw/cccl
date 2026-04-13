// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

// Benchmark for indirect (device-accessible num_items) radix sort pairs.
// Compares indirect stream launch, standard stream launch, and graph-captured replay.

#include <cub/device/device_radix_sort.cuh>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include <nvbench_helper.cuh>

// ============================================================================
// Stream-based benchmarks
// ============================================================================

template <typename KeyT, typename OffsetT>
void indirect_radix_sort_pairs(nvbench::state& state, nvbench::type_list<KeyT, OffsetT>)
try
{
  using offset_t = cub::detail::choose_offset_t<OffsetT>;
  using value_t  = nvbench::int32_t;

  const auto elements     = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const double fill_ratio = state.get_float64("FillRatio");
  const bit_entropy entropy = str_to_entropy(state.get_string("Entropy"));
  const auto actual_items = static_cast<offset_t>(elements * fill_ratio);
  const auto max_items    = static_cast<offset_t>(elements);

  constexpr int begin_bit = 0;
  constexpr int end_bit   = sizeof(KeyT) * 8;

  thrust::device_vector<KeyT> keys_in     = generate(elements, entropy);
  thrust::device_vector<KeyT> keys_out(elements);
  thrust::device_vector<value_t> values_in(elements);
  thrust::sequence(values_in.begin(), values_in.end());
  thrust::device_vector<value_t> values_out(elements);

  thrust::device_vector<offset_t> d_num_items_vec(1, actual_items);
  const offset_t* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

  state.add_element_count(actual_items);
  state.add_global_memory_reads<KeyT>(actual_items, "KeySize");
  state.add_global_memory_reads<value_t>(actual_items, "ValueSize");
  state.add_global_memory_writes<KeyT>(actual_items);
  state.add_global_memory_writes<value_t>(actual_items);

  size_t tmp_size{};
  cub::DeviceRadixSort::SortPairs(
    nullptr, tmp_size,
    thrust::raw_pointer_cast(keys_in.data()), thrust::raw_pointer_cast(keys_out.data()),
    thrust::raw_pointer_cast(values_in.data()), thrust::raw_pointer_cast(values_out.data()),
    d_num_items, max_items, begin_bit, end_bit);

  thrust::device_vector<nvbench::uint8_t> tmp(tmp_size, thrust::no_init);

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    cub::DeviceRadixSort::SortPairs(
      thrust::raw_pointer_cast(tmp.data()), tmp_size,
      thrust::raw_pointer_cast(keys_in.data()), thrust::raw_pointer_cast(keys_out.data()),
      thrust::raw_pointer_cast(values_in.data()), thrust::raw_pointer_cast(values_out.data()),
      d_num_items, max_items, begin_bit, end_bit, launch.get_stream());
  });
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

template <typename KeyT, typename OffsetT>
void standard_radix_sort_pairs(nvbench::state& state, nvbench::type_list<KeyT, OffsetT>)
try
{
  using offset_t = cub::detail::choose_offset_t<OffsetT>;
  using value_t  = nvbench::int32_t;

  const auto elements       = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const bit_entropy entropy = str_to_entropy(state.get_string("Entropy"));

  constexpr int begin_bit = 0;
  constexpr int end_bit   = sizeof(KeyT) * 8;

  thrust::device_vector<KeyT> keys_in     = generate(elements, entropy);
  thrust::device_vector<KeyT> keys_out(elements);
  thrust::device_vector<value_t> values_in(elements);
  thrust::sequence(values_in.begin(), values_in.end());
  thrust::device_vector<value_t> values_out(elements);

  state.add_element_count(elements);
  state.add_global_memory_reads<KeyT>(elements, "KeySize");
  state.add_global_memory_reads<value_t>(elements, "ValueSize");
  state.add_global_memory_writes<KeyT>(elements);
  state.add_global_memory_writes<value_t>(elements);

  size_t tmp_size{};
  cub::DeviceRadixSort::SortPairs(
    nullptr, tmp_size,
    thrust::raw_pointer_cast(keys_in.data()), thrust::raw_pointer_cast(keys_out.data()),
    thrust::raw_pointer_cast(values_in.data()), thrust::raw_pointer_cast(values_out.data()),
    static_cast<offset_t>(elements), begin_bit, end_bit);

  thrust::device_vector<nvbench::uint8_t> tmp(tmp_size, thrust::no_init);

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    cub::DeviceRadixSort::SortPairs(
      thrust::raw_pointer_cast(tmp.data()), tmp_size,
      thrust::raw_pointer_cast(keys_in.data()), thrust::raw_pointer_cast(keys_out.data()),
      thrust::raw_pointer_cast(values_in.data()), thrust::raw_pointer_cast(values_out.data()),
      static_cast<offset_t>(elements), begin_bit, end_bit, launch.get_stream());
  });
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

// ============================================================================
// CUDA graph benchmarks
// ============================================================================

template <typename KeyT, typename OffsetT>
void indirect_radix_sort_pairs_graph(nvbench::state& state, nvbench::type_list<KeyT, OffsetT>)
try
{
  using offset_t = cub::detail::choose_offset_t<OffsetT>;
  using value_t  = nvbench::int32_t;

  const auto elements     = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const double fill_ratio = state.get_float64("FillRatio");
  const bit_entropy entropy = str_to_entropy(state.get_string("Entropy"));
  const auto actual_items = static_cast<offset_t>(elements * fill_ratio);
  const auto max_items    = static_cast<offset_t>(elements);

  constexpr int begin_bit = 0;
  constexpr int end_bit   = sizeof(KeyT) * 8;

  thrust::device_vector<KeyT> keys_in     = generate(elements, entropy);
  thrust::device_vector<KeyT> keys_out(elements);
  thrust::device_vector<value_t> values_in(elements);
  thrust::sequence(values_in.begin(), values_in.end());
  thrust::device_vector<value_t> values_out(elements);

  thrust::device_vector<offset_t> d_num_items_vec(1, actual_items);
  const offset_t* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

  state.add_element_count(actual_items);
  state.add_global_memory_reads<KeyT>(actual_items, "KeySize");
  state.add_global_memory_reads<value_t>(actual_items, "ValueSize");
  state.add_global_memory_writes<KeyT>(actual_items);
  state.add_global_memory_writes<value_t>(actual_items);

  size_t tmp_size{};
  cub::DeviceRadixSort::SortPairs(
    nullptr, tmp_size,
    thrust::raw_pointer_cast(keys_in.data()), thrust::raw_pointer_cast(keys_out.data()),
    thrust::raw_pointer_cast(values_in.data()), thrust::raw_pointer_cast(values_out.data()),
    d_num_items, max_items, begin_bit, end_bit);

  thrust::device_vector<nvbench::uint8_t> tmp(tmp_size, thrust::no_init);

  // Capture graph once
  cudaStream_t capture_stream{};
  cudaStreamCreate(&capture_stream);

  cudaGraph_t graph{};
  cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal);

  cub::DeviceRadixSort::SortPairs(
    thrust::raw_pointer_cast(tmp.data()), tmp_size,
    thrust::raw_pointer_cast(keys_in.data()), thrust::raw_pointer_cast(keys_out.data()),
    thrust::raw_pointer_cast(values_in.data()), thrust::raw_pointer_cast(values_out.data()),
    d_num_items, max_items, begin_bit, end_bit, capture_stream);

  cudaStreamEndCapture(capture_stream, &graph);

  cudaGraphExec_t graph_exec{};
  cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);

  // Benchmark replaying the graph
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

// Stream-based indirect sort pairs
NVBENCH_BENCH_TYPES(indirect_radix_sort_pairs, NVBENCH_TYPE_AXES(sort_types, offset_types))
  .set_name("indirect_radix_sort_pairs")
  .set_type_axes_names({"KeyT{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(10, 28, 2))
  .add_float64_axis("FillRatio", {0.001, 0.01, 0.1, 0.5, 1.0})
  .add_string_axis("Entropy", {"1.000", "0.201"});

// Standard sort pairs baseline
NVBENCH_BENCH_TYPES(standard_radix_sort_pairs, NVBENCH_TYPE_AXES(sort_types, offset_types))
  .set_name("standard_radix_sort_pairs")
  .set_type_axes_names({"KeyT{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(10, 28, 2))
  .add_string_axis("Entropy", {"1.000", "0.201"});

// Graph-captured indirect sort pairs (the real use case!)
NVBENCH_BENCH_TYPES(indirect_radix_sort_pairs_graph, NVBENCH_TYPE_AXES(sort_types, offset_types))
  .set_name("indirect_radix_sort_pairs_graph")
  .set_type_axes_names({"KeyT{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(10, 28, 2))
  .add_float64_axis("FillRatio", {0.001, 0.01, 0.1, 0.5, 1.0})
  .add_string_axis("Entropy", {"1.000", "0.201"});
