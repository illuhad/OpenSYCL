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

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <chrono>
#include "bruteforce_nbody.hpp"
#include "model.hpp"


arithmetic_type mirror_position(const arithmetic_type mirror_pos,
                                const arithmetic_type position)
{
  arithmetic_type delta = sycl::fabs(mirror_pos - position);
  return (position <= mirror_pos) ?
        mirror_pos + delta : mirror_pos - delta;
}

int get_num_iterations_per_output_step()
{
  char* val = std::getenv("NBODY_ITERATIONS_PER_OUTPUT");
  if(!val)
    return 10;
  return std::stoi(val);
}


int main()
{
  const int iterations_per_output =
      get_num_iterations_per_output_step();

  std::vector<particle_type> particles;
  std::vector<vector_type> velocities;

  arithmetic_type particle_mass = total_mass / num_particles;

  random_particle_cloud particle_distribution0{
    vector_type{0.0f, 100.0f, 0.0f},
    vector_type{10.0f, 15.0f, 11.0f},
    particle_mass, 0.1f * particle_mass,
    vector_type{0.0f, -26.0f, 5.0f},
    vector_type{5.0f, 20.0f, 12.f}
  };


  random_particle_cloud particle_distribution1{
    vector_type{50.0f, 5.0f, 0.0f},
    vector_type{17.0f, 7.0f, 5.0f},
    particle_mass, 0.1f * particle_mass,
    vector_type{-5.f, 20.0f, 1.0f},
    vector_type{18.0f, 11.f, 8.f}
  };

  random_particle_cloud particle_distribution2{
    vector_type{-50.0f, -100.0f, 0.0f},
    vector_type{10.0f, 10.0f, 14.0f},
    particle_mass, 0.1f * particle_mass,
    vector_type{5.f, 5.0f, -1.0f},
    vector_type{10.0f, 6.f, 5.f}
  };

  particle_distribution0.sample(0.2 * num_particles,
                                particles, velocities);

  std::vector<particle_type> particles_cloud1, particles_cloud2;
  std::vector<vector_type> velocities_cloud1, velocities_cloud2;
  particle_distribution1.sample(0.4 * num_particles,
                                particles_cloud1, velocities_cloud1);

  particle_distribution2.sample(0.4 * num_particles,
                                particles_cloud2, velocities_cloud2);

  particles.insert(particles.end(),
                   particles_cloud1.begin(),
                   particles_cloud1.end());

  particles.insert(particles.end(),
                   particles_cloud2.begin(),
                   particles_cloud2.end());

  velocities.insert(velocities.end(),
                   velocities_cloud1.begin(),
                   velocities_cloud1.end());

  velocities.insert(velocities.end(),
                   velocities_cloud2.begin(),
                   velocities_cloud2.end());

  sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};
  
  particle_type* particles_buffer = sycl::malloc_device<particle_type>(particles.size(), q);
  vector_type* velocities_buffer = sycl::malloc_device<vector_type>(velocities.size(), q);
  vector_type* forces_buffer = sycl::malloc_device<vector_type>(particles.size(), q);

  q.copy(particles.data(), particles_buffer, particles.size());
  q.copy(velocities.data(), velocities_buffer, particles.size());
  
  auto execution_range = sycl::nd_range<1>{
      sycl::range<1>{((num_particles + local_size - 1) / local_size) * local_size},
      sycl::range<1>{local_size}
  };


  std::ofstream outputfile{"output.txt"};

  const std::size_t num_particles = particles.size();

  auto start_time = std::chrono::high_resolution_clock::now();
  double total_time = 0.0;

  for(std::size_t t = 0; t < num_timesteps; ++t)
  {
    // Submit force calculation
    q.submit([&](sycl::handler& cgh){

      auto scratch = sycl::local_accessor<particle_type, 1>{
        sycl::range<1>{local_size},
        cgh
      };

      cgh.parallel_for(execution_range,
                      [=](sycl::nd_item<1> tid){
        const std::size_t global_id = tid.get_global_id().get(0);
        const std::size_t local_id = tid.get_local_id().get(0);
        
        vector_type force{0.0f};

        const particle_type my_particle =
            (global_id < num_particles) ? particles_buffer[global_id] : particle_type{0.0f};

        for(size_t offset = 0; offset < num_particles; offset += local_size)
        {
          if(offset + local_id < num_particles)
            scratch[local_id] = particles_buffer[offset + local_id];
          else
            scratch[local_id] = particle_type{0.0f};

          sycl::group_barrier(tid.get_group());

          for(int i = 0; i < local_size; ++i)
          {
            const particle_type p = scratch[i];
            const vector_type p_direction = p.swizzle<0,1,2>();
            // 3 flops
            const vector_type R = p_direction - my_particle.swizzle<0,1,2>();
            
            // 6 flops (ignoring rsqrt, where we cannot quantify - this
            //   will be a major source of the reported number being off
            //   from peak)
            const arithmetic_type r_inv =
                sycl::rsqrt(R.x()*R.x() + R.y()*R.y() + R.z()*R.z()
                                    + gravitational_softening);

            // Actually we just calculate the acceleration, not the
            // force. We only need the acceleration anyway.
            if(global_id != offset + i)
              // 9 flops
              force += static_cast<arithmetic_type>(p.w()) * r_inv * r_inv * r_inv * R;
          }

          sycl::group_barrier(tid.get_group());
        }

        if(global_id < num_particles)
          forces_buffer[global_id] = force;
      });
    });

    // Time integration
    q.parallel_for(execution_range,
                   [=](sycl::nd_item<1> tid){
      const size_t global_id = tid.get_global_id().get(0);

      if(global_id < num_particles)
      {
        particle_type p = particles_buffer[global_id];
        vector_type v = velocities_buffer[global_id];
        const vector_type acceleration = forces_buffer[global_id];

        // Bring v to the current state
        v += acceleration * dt;

        // Update position
        p.x() += v.x() * dt;
        p.y() += v.y() * dt;
        p.z() += v.z() * dt;

        // Reflect particle position and invert velocities
        // if particles exit the simulation cube
        if(static_cast<arithmetic_type>(p.x()) <= -half_cube_size)
        {
          v.x() = sycl::fabs(v.x());
          p.x() = mirror_position(-half_cube_size, p.x());
        }
        else if(static_cast<arithmetic_type>(p.x()) >= half_cube_size)
        {
          v.x() = -sycl::fabs(v.x());
          p.x() = mirror_position(half_cube_size, p.x());
        }

        if(static_cast<arithmetic_type>(p.y()) <= -half_cube_size)
        {
          v.y() = sycl::fabs(v.y());
          p.y() = mirror_position(-half_cube_size, p.y());
        }
        else if(static_cast<arithmetic_type>(p.y()) >= half_cube_size)
        {
          v.y() = -sycl::fabs(v.y());
          p.y() = mirror_position(half_cube_size, p.y());
        }

        if(static_cast<arithmetic_type>(p.z()) <= -half_cube_size)
        {
          v.z() = sycl::fabs(v.z());
          p.z() = mirror_position(-half_cube_size, p.z());
        }
        else if(static_cast<arithmetic_type>(p.z()) >= half_cube_size)
        {
          v.z() = -sycl::fabs(v.z());
          p.z() = mirror_position(half_cube_size, p.z());
        }

        particles_buffer[global_id] = p;
        velocities_buffer[global_id] = v;
      }
    });
  

    if(t % iterations_per_output == 0)
    {
      // This wait is only needed for the performance measurement.
      // We don't need it for the algorithm itself - but we don't want
      // to include the data transfer time in the measurement.
      q.wait();
      auto stop_time = std::chrono::high_resolution_clock::now();
      total_time +=
          std::chrono::duration_cast<std::chrono::nanoseconds>(stop_time - start_time)
              .count() *
          1.e-9;
      
      const std::size_t flops_per_iter =
          18 * num_particles * num_particles + 12 * num_particles;
      std::cout << "Overall average performance: "
                << 1.e-9 * flops_per_iter * (t + 1) / total_time << " GFlops"
                << std::endl;

      q.copy(particles_buffer, particles.data(), particles.size()).wait();

      std::cout << "Writing output..."  << std::endl;
      for(std::size_t i = 0; i < num_particles; ++i)
      {
        outputfile << particles[i].x() << " "
                   << particles[i].y() << " "
                   << particles[i].z() << " " << i << std::endl;
      }

      // start again for next iteration
      start_time = std::chrono::high_resolution_clock::now();
    }
  }

  q.wait();
  sycl::free(particles_buffer, q);
  sycl::free(velocities_buffer, q);
  sycl::free(forces_buffer, q);
}
