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

/** Built-in flags **/
static constexpr uint32_t BASE_UNIT_SIZE = 32;
static constexpr uint32_t EMPTY_SLAB_POINTER = 0xFFFFFFFF;
static constexpr uint32_t EMPTY_KEY = 0xFFFFFFFF;
static constexpr uint32_t EMPTY_VALUE = 0xFFFFFFFF;
static constexpr uint64_t EMPTY_PAIR_64 = 0xFFFFFFFFFFFFFFFFLL;

/** Queries **/
static constexpr uint32_t SEARCH_NOT_FOUND = 0xFFFFFFFF;

/** Warp operations **/
static constexpr uint32_t WARP_WIDTH = 32;

static constexpr uint32_t HEAD_SLAB_POINTER = 0xFFFFFFFE;
static constexpr uint32_t ACTIVE_LANE_MASK = 0xFFFFFFFF;
static constexpr uint32_t NEXT_SLAB_POINTER_LANE = 31;

/* bits:   31 | 30 | ... | 3 | 2 | 1 | 0 */
/* keys:    0    0         0   1   0   1 => 00 01 01 01 ... 01 01 */
/* data:    0    0         1   1   1   1 => 00 11 11 11 ... 11 11 */
static constexpr uint32_t REGULAR_NODE_KEY_MASK = 0x15555555;
static constexpr uint32_t REGULAR_NODE_DATA_MASK = 0x3FFFFFFF;
