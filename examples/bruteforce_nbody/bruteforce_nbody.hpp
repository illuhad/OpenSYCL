/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause

#ifndef BRUTEFORCE_NBODY_HPP
#define BRUTEFORCE_NBODY_HPP

#include <sycl/sycl.hpp>

using arithmetic_type = float;
using vector_type = sycl::vec<arithmetic_type, 3>;
using particle_type = sycl::vec<arithmetic_type, 4>;


constexpr arithmetic_type total_mass = 100000.f;
constexpr std::size_t num_particles = 40000;
constexpr std::size_t num_timesteps = 600;


constexpr std::size_t local_size = 128;
constexpr arithmetic_type gravitational_softening = 1.e-4f;
constexpr arithmetic_type dt = 0.1f;

constexpr arithmetic_type cube_size = 400.0f;
constexpr arithmetic_type half_cube_size = 0.5f * cube_size;

#endif
