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

#include <boost/test/tools/old/interface.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/test/unit_test_suite.hpp>

#include <cstdint>
#include <random>
#include <unordered_set>
#include <thread>
#include <hipSYCL/std/stdpar/detail/allocation_map.hpp>


#include "pstl_test_suite.hpp"

BOOST_AUTO_TEST_SUITE(pstl_allocation_map)

struct payload{};
using amap_t = hipsycl::common::allocation_map<payload>;

template<class F>
void for_each_test_allocation(std::size_t n, F&& f) {
  std::mt19937 gen(123);
  std::uniform_int_distribution<uint64_t> dist;

  for(std::size_t i = 0; i < n; ++i) {
    f(dist(gen));
  }
}

BOOST_AUTO_TEST_CASE(insert) {
  std::unordered_set<uint64_t> addresses;
  for_each_test_allocation(50000, [&](uint64_t x){
    addresses.insert(x);
  });
  if(addresses.find(0) == addresses.end())
    addresses.insert(0);
  // ~0 is unsupported, due to overflow for non-zero allocation ranges.
  if(addresses.find(~0ull - 1 ) == addresses.end())
    addresses.insert(~0ull - 1);

  amap_t amap;
  for(const auto& x : addresses) {
    amap_t::value_type v;
    v.allocation_size = 1;
    BOOST_CHECK(amap.insert(x, v));
  }

  for(const auto& x : addresses) {
    uint64_t root_address = 0;
    auto* ret = amap.get_entry(x, root_address);
    BOOST_CHECK(ret != nullptr);
    if(ret) {
      BOOST_CHECK(root_address == x);
      BOOST_CHECK(ret->allocation_size == 1);
    }
  }

  std::mt19937 gen(222);
  std::uniform_int_distribution<uint64_t> dist;
  
  for(int num_foreign_addresses_found = 0; num_foreign_addresses_found < 10000;) {
    uint64_t addr = dist(gen);
    if(addresses.find(addr) == addresses.end()) {
      uint64_t root_address = 0;
      auto* ret = amap.get_entry(addr, root_address);
      BOOST_CHECK(!ret);
      ++num_foreign_addresses_found;
    }
  }
}

BOOST_AUTO_TEST_CASE(get_entry) {
  amap_t amap;

  std::unordered_map<uint64_t, uint64_t> reference_entries;

  uint64_t current_address = 0;
  for(uint64_t i = 1; i < 1000; ++i) {

    amap_t::value_type v;
    v.allocation_size = i;

    BOOST_CHECK(amap.insert(current_address, v));
    reference_entries[current_address] = i;

    current_address += i;
  }

  for(const auto& entry : reference_entries) {
    uint64_t root_address = 0;
    auto* ret = amap.get_entry(entry.first, root_address);
    BOOST_CHECK(ret);
    if(ret) {
      BOOST_CHECK(root_address == entry.first);
      BOOST_CHECK(ret->allocation_size == entry.second);
    }
    root_address = 0;
    ret = amap.get_entry(entry.first + entry.second / 2, root_address);

    BOOST_CHECK(ret);
    if(ret) {
      BOOST_CHECK(root_address == entry.first);
      BOOST_CHECK(ret->allocation_size == entry.second);
    }

    root_address = 0;
    ret = amap.get_entry(entry.first + entry.second - 1, root_address);

    BOOST_CHECK(ret);
    if(ret) {
      BOOST_CHECK(root_address == entry.first);
      BOOST_CHECK(ret->allocation_size == entry.second);
    }
  }
}

BOOST_AUTO_TEST_CASE(erase) {
  
  std::unordered_set<uint64_t> addresses;
  for_each_test_allocation(50000, [&](uint64_t x){
    addresses.insert(x);
  });
  if(addresses.find(0) == addresses.end())
    addresses.insert(0);
  // ~0 is unsupported, due to overflow for non-zero allocation ranges.
  if(addresses.find(~0ull - 1 ) == addresses.end())
    addresses.insert(~0ull - 1);

  amap_t amap;
  for(const auto& x : addresses) {
    amap_t::value_type v;
    v.allocation_size = 1;
    BOOST_CHECK(amap.insert(x, v));
  }

  for(const auto& x : addresses) {
    uint64_t root_address = 0;
    BOOST_CHECK(amap.get_entry(x, root_address));
    BOOST_CHECK(amap.erase(x));
    BOOST_CHECK(!amap.get_entry(x, root_address));
  }
}

BOOST_AUTO_TEST_CASE(multi_threaded) {
  const std::size_t num_threads = 4;
  const std::size_t elements_per_thread = 2000;
  std::vector<std::thread> threads;

  std::vector<std::pair<uint64_t, uint64_t>> reference_entries;

  uint64_t current_address = 0;
  for(uint64_t i = 1; i < elements_per_thread*num_threads+1; ++i) {
    reference_entries.push_back(std::make_pair(current_address, i));
    current_address += i;    
  }

  amap_t amap;
  std::vector<int> per_thread_num_successes(num_threads, 0);

  for(int thread_id = 0; thread_id < num_threads; ++thread_id) {
    threads.push_back(std::thread{
        [=, &amap, &reference_entries, &per_thread_num_successes]() {
          int &num_successes = per_thread_num_successes[thread_id];
          {
            bool res = true;

            for (int j = 0; j < elements_per_thread; ++j) {
              int elem = j * num_threads + thread_id;
              amap_t::value_type v;
              v.allocation_size = reference_entries[elem].second;
              if(!amap.insert(reference_entries[elem].first, v)) {
                res = false;
              }
            }

            if (res)
              num_successes++;
          }
          {
            bool res = true;

            for(int j = 0; j < elements_per_thread/2; ++j) {
              int elem = j * num_threads + thread_id;
              if(!amap.erase(reference_entries[elem].first)) {
                res = false;
              } else {
                uint64_t root_address = 0;
                auto* query_result = amap.get_entry(reference_entries[elem].first, root_address);
                if(query_result != nullptr) {
                  res = false;
                }
              }
            }

            if(res)
              num_successes++;
          }
          {
            bool res = true;

            for(int j = elements_per_thread/2; j < elements_per_thread; ++j) {
              int elem = j * num_threads + thread_id;

              uint64_t root_address = 0;
              auto* query_result = amap.get_entry(reference_entries[elem].first, root_address);
              if(!query_result) {
                res = false;
              } else {
                if(root_address != reference_entries[elem].first) {
                  res = false;
                }
                if(query_result->allocation_size != reference_entries[elem].second) {
                  res = false;
                }
              }

              if(!amap.erase(reference_entries[elem].first))
                res = false;
            }

            if(res)
              num_successes++;
          }
        }});
  }

  for(int i = 0; i < num_threads; ++i)
    threads[i].join();

  for(int i = 0; i < num_threads; ++i) {
    BOOST_TEST(per_thread_num_successes[i] == 3);
  }
}


BOOST_AUTO_TEST_SUITE_END()
