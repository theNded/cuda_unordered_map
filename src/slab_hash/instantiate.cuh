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

#include <cassert>
#include <iostream>
#include <random>
#include <typeinfo>

// global declarations
#include "config.cuh"

// global helper methods:
#include "helper/helper_cuda.h"
#include "helper_warp.cuh"

// class declaration:
#include "iterator.cuh"
#include "slab_hash/concurrent_map/cmap.cuh"
#include "slab_hash/concurrent_set/cset.cuh"

// warp implementations of member functions:
#include "slab_hash/concurrent_map/cmap_device.cuh"
#include "slab_hash/concurrent_set/cset_device.cuh"

#include "slab_hash/concurrent_map/cmap_kernel.cuh"
#include "slab_hash/concurrent_set/cset_kernel.cuh"

// implementations:
#include "slab_hash/concurrent_map/cmap_host.cuh"
#include "slab_hash/concurrent_set/cset_host.cuh"