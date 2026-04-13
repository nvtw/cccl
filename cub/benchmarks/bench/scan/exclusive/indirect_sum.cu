// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

// Benchmark for indirect (device-accessible num_items) scan vs standard scan.
// Includes CUDA graph capture benchmarks to measure the real use case:
// capture once, replay many times with varying lengths.

#include <cub/device/device_scan.cuh>

#include <thrust/device_vector.h>

#include <nvbench_helper.cuh>

// ============================================================================
// Stream-based benchmarks (direct kernel launch)
// ============================================================================

template <typename T, typename OffsetT>
void indirect_exclusive_sum(nvbench::state& state, nvbench::type_list<T, OffsetT>)
try
{
  using offset_t = cub::detail::choose_offset_t<OffsetT>;

  const auto elements     = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const double fill_ratio = state.get_float64("FillRatio");
  const auto actual_items = static_cast<offset_t>(elements * fill_ratio);
  const auto max_items    = static_cast<offset_t>(elements);

  thrust::device_vector<T> input = generate(elements);
  thrust::device_vector<T> output(elements);

  thrust::device_vector<offset_t> d_num_items_vec(1, actual_items);
  const offset_t* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

  state.add_element_count(actual_items);
  state.add_global_memory_reads<T>(actual_items, "Size");
  state.add_global_memory_writes<T>(actual_items);

  size_t tmp_size{};
  cub::DeviceScan::ExclusiveSum(
    nullptr,
    tmp_size,
    thrust::raw_pointer_cast(input.data()),
    thrust::raw_pointer_cast(output.data()),
    d_num_items,
    max_items);

  thrust::device_vector<nvbench::uint8_t> tmp(tmp_size, thrust::no_init);

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    cub::DeviceScan::ExclusiveSum(
      thrust::raw_pointer_cast(tmp.data()),
      tmp_size,
      thrust::raw_pointer_cast(input.data()),
      thrust::raw_pointer_cast(output.data()),
      d_num_items,
      max_items,
      launch.get_stream());
  });
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

template <typename T, typename OffsetT>
void standard_exclusive_sum(nvbench::state& state, nvbench::type_list<T, OffsetT>)
try
{
  using offset_t = cub::detail::choose_offset_t<OffsetT>;

  const auto elements = static_cast<std::size_t>(state.get_int64("Elements{io}"));

  thrust::device_vector<T> input = generate(elements);
  thrust::device_vector<T> output(elements);

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(elements);

  size_t tmp_size{};
  cub::DeviceScan::ExclusiveSum(
    nullptr,
    tmp_size,
    thrust::raw_pointer_cast(input.data()),
    thrust::raw_pointer_cast(output.data()),
    static_cast<offset_t>(elements));

  thrust::device_vector<nvbench::uint8_t> tmp(tmp_size, thrust::no_init);

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    cub::DeviceScan::ExclusiveSum(
      thrust::raw_pointer_cast(tmp.data()),
      tmp_size,
      thrust::raw_pointer_cast(input.data()),
      thrust::raw_pointer_cast(output.data()),
      static_cast<offset_t>(elements),
      launch.get_stream());
  });
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

// ============================================================================
// CUDA graph benchmarks - captures once, replays (this is the real use case)
// ============================================================================

template <typename T, typename OffsetT>
void indirect_exclusive_sum_graph(nvbench::state& state, nvbench::type_list<T, OffsetT>)
try
{
  using offset_t = cub::detail::choose_offset_t<OffsetT>;

  const auto elements     = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const double fill_ratio = state.get_float64("FillRatio");
  const auto actual_items = static_cast<offset_t>(elements * fill_ratio);
  const auto max_items    = static_cast<offset_t>(elements);

  thrust::device_vector<T> input = generate(elements);
  thrust::device_vector<T> output(elements);

  thrust::device_vector<offset_t> d_num_items_vec(1, actual_items);
  const offset_t* d_num_items = thrust::raw_pointer_cast(d_num_items_vec.data());

  state.add_element_count(actual_items);
  state.add_global_memory_reads<T>(actual_items, "Size");
  state.add_global_memory_writes<T>(actual_items);

  size_t tmp_size{};
  cub::DeviceScan::ExclusiveSum(
    nullptr,
    tmp_size,
    thrust::raw_pointer_cast(input.data()),
    thrust::raw_pointer_cast(output.data()),
    d_num_items,
    max_items);

  thrust::device_vector<nvbench::uint8_t> tmp(tmp_size, thrust::no_init);

  // Capture graph once
  cudaStream_t capture_stream{};
  cudaStreamCreate(&capture_stream);

  cudaGraph_t graph{};
  cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal);

  cub::DeviceScan::ExclusiveSum(
    thrust::raw_pointer_cast(tmp.data()),
    tmp_size,
    thrust::raw_pointer_cast(input.data()),
    thrust::raw_pointer_cast(output.data()),
    d_num_items,
    max_items,
    capture_stream);

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

using types = nvbench::type_list<nvbench::int32_t, nvbench::float32_t>;

// Stream-based indirect scan at various fill ratios
NVBENCH_BENCH_TYPES(indirect_exclusive_sum, NVBENCH_TYPE_AXES(types, offset_types))
  .set_name("indirect_exclusive_sum")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(10, 28, 2))
  .add_float64_axis("FillRatio", {0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1.0});

// Standard scan baseline
NVBENCH_BENCH_TYPES(standard_exclusive_sum, NVBENCH_TYPE_AXES(types, offset_types))
  .set_name("standard_exclusive_sum")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(10, 28, 2));

// Graph-captured indirect scan (the real use case)
NVBENCH_BENCH_TYPES(indirect_exclusive_sum_graph, NVBENCH_TYPE_AXES(types, offset_types))
  .set_name("indirect_exclusive_sum_graph")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(10, 28, 2))
  .add_float64_axis("FillRatio", {0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1.0});
