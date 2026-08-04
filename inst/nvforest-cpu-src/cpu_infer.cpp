/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Derived from nvForest 26.06.00 cpu inference instantiations.
 */
#include <nvforest/detail/infer/cpu.hpp>
#include <nvforest/detail/specializations/infer_macros.hpp>

namespace nvforest::detail::inference {
NVFOREST_INFER_ALL(template, raft_proto::device_type::cpu, 0)
NVFOREST_INFER_ALL(template, raft_proto::device_type::cpu, 1)
NVFOREST_INFER_ALL(template, raft_proto::device_type::cpu, 2)
NVFOREST_INFER_ALL(template, raft_proto::device_type::cpu, 3)
NVFOREST_INFER_ALL(template, raft_proto::device_type::cpu, 4)
NVFOREST_INFER_ALL(template, raft_proto::device_type::cpu, 5)
NVFOREST_INFER_ALL(template, raft_proto::device_type::cpu, 6)
NVFOREST_INFER_ALL(template, raft_proto::device_type::cpu, 7)
NVFOREST_INFER_ALL(template, raft_proto::device_type::cpu, 8)
NVFOREST_INFER_ALL(template, raft_proto::device_type::cpu, 9)
NVFOREST_INFER_ALL(template, raft_proto::device_type::cpu, 10)
NVFOREST_INFER_ALL(template, raft_proto::device_type::cpu, 11)
} // namespace nvforest::detail::inference
