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

#include "sycl_test_suite.hpp"
#include <boost/test/tools/old/interface.hpp>

BOOST_FIXTURE_TEST_SUITE(profiler_tests, reset_device_fixture)

bool default_device_supports_profiling() {
  cl::sycl::queue q;
  return q.get_device().get_info<cl::sycl::info::device::queue_profiling>();
}

BOOST_AUTO_TEST_CASE(queue_no_profiling_exception)
{
  const auto is_invalid = [](const cl::sycl::exception &e) {
    return e.code() ==  cl::sycl::errc::invalid;
  };

  cl::sycl::queue queue;  // no enable_profiling

  cl::sycl::buffer<int, 1> buf1{cl::sycl::range<1>(1)};
  auto evt1 = queue.submit([&](cl::sycl::handler &cgh) {
    auto acc = buf1.get_access<cl::sycl::access::mode::discard_write>(cgh);
    cgh.single_task<class no_profile_single_task>([=]() {
      acc[0] = 42;
    });
  });
  BOOST_CHECK_EXCEPTION(evt1.get_profiling_info<cl::sycl::info::event_profiling::command_submit>(),
                        cl::sycl::exception,
                        is_invalid);

  auto evt2 = queue.submit([&](cl::sycl::handler &cgh) {
    auto acc = buf1.get_access<cl::sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<class no_profile_parallel_for>(buf1.get_range(), [=](cl::sycl::item<1> item) {
      acc[item] = item.get_id()[0];
    });
  });

  cl::sycl::buffer<int, 1> buf2{buf1.get_range()};
  auto evt3 = queue.submit([&](cl::sycl::handler &cgh) {
    cgh.copy(buf1.get_access<cl::sycl::access::mode::read>(cgh),
             buf2.get_access<cl::sycl::access::mode::discard_write>(cgh));
  });

  BOOST_CHECK_EXCEPTION(evt3.get_profiling_info<cl::sycl::info::event_profiling::command_end>(),
                        cl::sycl::exception,
                        is_invalid);
  BOOST_CHECK_EXCEPTION(evt2.get_profiling_info<cl::sycl::info::event_profiling::command_start>(),
                        cl::sycl::exception,
                        is_invalid);

  auto evt4 = queue.submit([&](cl::sycl::handler &cgh) {
    cgh.fill(buf1.get_access<cl::sycl::access::mode::discard_write>(cgh), 1);
  });

  int host_array;
  auto evt5 = queue.submit([&](cl::sycl::handler &cgh) {
    // copy to pointer
    cgh.copy(buf2.get_access<cl::sycl::access::mode::read>(cgh), &host_array);
  });
  
  BOOST_CHECK_EXCEPTION(evt4.get_profiling_info<cl::sycl::info::event_profiling::command_submit>(),
                        cl::sycl::exception,
                        is_invalid);
  BOOST_CHECK_EXCEPTION(evt5.get_profiling_info<cl::sycl::info::event_profiling::command_submit>(),
                        cl::sycl::exception,
                        is_invalid);
  queue.wait();
}

BOOST_AUTO_TEST_CASE(queue_profiling)
{
  if(default_device_supports_profiling()) {
    cl::sycl::queue queue{cl::sycl::property::queue::enable_profiling{}};
    constexpr size_t n = 4;
    cl::sycl::buffer<int, 1> buf1{cl::sycl::range<1>(n)};

    auto evt1 = queue.submit([&](cl::sycl::handler &cgh) {
      auto acc = buf1.get_access<cl::sycl::access::mode::discard_write>(cgh);
      cgh.single_task<class profile_single_task>([=]() { acc[0] = 42; });
    });

    auto t12 = evt1.get_profiling_info<
        cl::sycl::info::event_profiling::command_start>();
    auto t11 = evt1.get_profiling_info<
        cl::sycl::info::event_profiling::command_submit>();
    auto t13 =
        evt1.get_profiling_info<cl::sycl::info::event_profiling::command_end>();
    // We cannot test that submit time is <= command start time, since
    // in some backends (e.g. CUDA) time is only measured as elapsed time
    // between two points in low-precision float. Submission time on the other
    // hand is always exact. So there might be rounding errors causing
    // t12 > t11.
    // The same thing could in principle happen when comparing submission time
    // with command end time, but hopefully this is less likely.
    BOOST_CHECK(t11 <= t13 && t12 <= t13);

    auto evt2 = queue.submit([&](cl::sycl::handler &cgh) {
      auto acc = buf1.get_access<cl::sycl::access::mode::discard_write>(cgh);
      cgh.parallel_for<class profile_parallel_for>(
          buf1.get_range(),
          [=](cl::sycl::item<1> item) { acc[item] = item.get_id()[0]; });
    });

    cl::sycl::buffer<int, 1> buf2{buf1.get_range()};
    auto evt3 = queue.submit([&](cl::sycl::handler &cgh) {
      cgh.copy(buf1.get_access<cl::sycl::access::mode::read>(cgh),
               buf2.get_access<cl::sycl::access::mode::discard_write>(cgh));
    });

    auto t21 = evt2.get_profiling_info<
        cl::sycl::info::event_profiling::command_submit>();
    auto t22 = evt2.get_profiling_info<
        cl::sycl::info::event_profiling::command_start>();
    auto t23 =
        evt2.get_profiling_info<cl::sycl::info::event_profiling::command_end>();
    BOOST_CHECK(t21 <= t23 && t22 <= t23);

    auto t31 = evt3.get_profiling_info<
        cl::sycl::info::event_profiling::command_submit>();
    auto t32 = evt3.get_profiling_info<
        cl::sycl::info::event_profiling::command_start>();
    auto t33 =
        evt3.get_profiling_info<cl::sycl::info::event_profiling::command_end>();
    BOOST_CHECK(t31 <= t33 && t32 <= t33);
    BOOST_CHECK(t21 <= t31 && t23 <= t32);

    auto evt4 = queue.submit([&](cl::sycl::handler &cgh) {
      cgh.fill(buf1.get_access<cl::sycl::access::mode::discard_write>(cgh), 1);
    });

    int host_array[n];
    auto evt5 = queue.submit([&](cl::sycl::handler &cgh) {
      // copy to pointer
      cgh.copy(buf2.get_access<cl::sycl::access::mode::read>(cgh), host_array);
    });

    auto t51 = evt5.get_profiling_info<
        cl::sycl::info::event_profiling::command_submit>();
    auto t52 = evt5.get_profiling_info<
        cl::sycl::info::event_profiling::command_start>();
    auto t53 =
        evt5.get_profiling_info<cl::sycl::info::event_profiling::command_end>();
    BOOST_CHECK(t51 <= t53 && t52 <= t53);

    // re-ordered
    auto t41 = evt4.get_profiling_info<
        cl::sycl::info::event_profiling::command_submit>();
    auto t42 = evt4.get_profiling_info<
        cl::sycl::info::event_profiling::command_start>();
    auto t43 =
        evt4.get_profiling_info<cl::sycl::info::event_profiling::command_end>();
    BOOST_CHECK(t41 <= t43 && t42 <= t43);

    // usm
    auto *src = cl::sycl::malloc_shared<int>(n, queue);
    auto *dest = cl::sycl::malloc_shared<int>(n, queue);
    auto evt6 = queue.submit(
        [&](cl::sycl::handler &cgh) { cgh.memset(src, 0, sizeof src); });

    auto t61 = evt6.get_profiling_info<
        cl::sycl::info::event_profiling::command_submit>();
    auto t62 = evt6.get_profiling_info<
        cl::sycl::info::event_profiling::command_start>();
    auto t63 =
        evt6.get_profiling_info<cl::sycl::info::event_profiling::command_end>();
    BOOST_CHECK(t61 <= t63 && t62 <= t63);

    auto evt7 = queue.submit(
        [&](cl::sycl::handler &cgh) { cgh.memcpy(dest, src, sizeof src); });
    auto t71 = evt7.get_profiling_info<
        cl::sycl::info::event_profiling::command_submit>();
    auto t72 = evt7.get_profiling_info<
        cl::sycl::info::event_profiling::command_start>();
    auto t73 =
        evt7.get_profiling_info<cl::sycl::info::event_profiling::command_end>();
    BOOST_CHECK(t71 <= t73 && t72 <= t73);

    auto evt8 = queue.submit(
        [&](cl::sycl::handler &cgh) { cgh.prefetch(dest, sizeof src); });
    auto t81 = evt8.get_profiling_info<
        cl::sycl::info::event_profiling::command_submit>();
    auto t82 = evt8.get_profiling_info<
        cl::sycl::info::event_profiling::command_start>();
    auto t83 =
        evt8.get_profiling_info<cl::sycl::info::event_profiling::command_end>();
    // run time may be zero if prefetching is a no-op
    BOOST_CHECK(t81 <= t83 && t82 <= t83);

    cl::sycl::free(src, queue);
    cl::sycl::free(dest, queue);
  }
}

// update_host is an explicit operation, but implemented as a requirement and thus not profiled.
// remove the warning and merge this test into queue_profiling if that is ever resolved.
BOOST_AUTO_TEST_CASE(queue_profiling_update_host_supported)
{
  if(default_device_supports_profiling()) {
    cl::sycl::queue queue{cl::sycl::property::queue::enable_profiling{}};
    int i = 1337;
    cl::sycl::buffer<int, 1> buf{&i, cl::sycl::range<1>(1)};
    queue.submit([&](cl::sycl::handler &cgh) {
      auto acc = buf.get_access<cl::sycl::access::mode::discard_write>(cgh);
      cgh.single_task<class basic_single_task>([=]() { acc[0] = 42; });
    });
    auto evt = queue.submit([&](cl::sycl::handler &cgh) {
      cgh.update_host(buf.get_access<cl::sycl::access::mode::read>(cgh));
    });
    BOOST_WARN_NO_THROW(evt.get_profiling_info<
                        cl::sycl::info::event_profiling::command_start>());
  }
}


BOOST_AUTO_TEST_SUITE_END() // NOTE: Make sure not to add anything below this line
