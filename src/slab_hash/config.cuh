/*
 * Copyright 2019 Saman Ashkiani
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied. See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "slab_alloc/slab_alloc.cuh"

// internal parameters for slab hash device functions:
static constexpr uint32_t EMPTY_KEY = 0xFFFFFFFF;
static constexpr uint32_t EMPTY_VALUE = 0xFFFFFFFF;
static constexpr uint64_t EMPTY_PAIR_64 = 0xFFFFFFFFFFFFFFFFLL;

static constexpr uint32_t WARP_WIDTH = 32;
static constexpr uint32_t SEARCH_NOT_FOUND = 0xFFFFFFFF;
static constexpr uint32_t ACTIVE_LANE_MASK = 0xFFFFFFFF;

static constexpr uint32_t HEAD_SLAB_POINTER = 0xFFFFFFFE;
static constexpr uint32_t EMPTY_INDEX_POINTER = 0xFFFFFFFF;

static constexpr uint32_t BASE_UNIT_SIZE = 32;
static constexpr uint32_t REGULAR_NODE_ADDRESS_MASK = 0x30000000;
static constexpr uint32_t REGULAR_NODE_DATA_MASK = 0x3FFFFFFF;
static constexpr uint32_t REGULAR_NODE_KEY_MASK = 0x15555555;