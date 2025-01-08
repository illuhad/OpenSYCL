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

#include "../sycl_test_suite.hpp"
#include "group_functions.hpp"

#ifdef HIPSYCL_ENABLE_GROUP_ALGORITHM_TESTS

BOOST_FIXTURE_TEST_SUITE(group_functions_tests, reset_device_fixture)

BOOST_AUTO_TEST_CASE_TEMPLATE(group_exclusive_scan_mul, T, test_types) {
  const size_t elements_per_thread = 1;
  const auto   data_generator      = [](std::vector<T> &v, size_t local_size,
                                 size_t global_size) {
    for (size_t i = 0; i < global_size; ++i)
      v[i] = (i < 1) ? T{static_cast<T>(2)} : T{static_cast<T>(1)};
  };

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      acc[global_linear_id] = sycl::exclusive_scan_over_group(
          g, local_value, detail::initialize_type<T>(10), std::multiplies<T>());
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig,size_t, size_t local_size,
                                        size_t global_size) {
      std::vector<T> expected(vOrig.size());

      for (size_t i = 0; i < global_size / local_size; ++i) {
        expected[i * local_size] = detail::initialize_type<T>(10);
        for (size_t j = 1; j < local_size; ++j)
          expected[i * local_size + j] =
              expected[i * local_size + j - 1] * vOrig[i * local_size + j - 1];

        for (size_t j = i * local_size; j < (i + 1) * local_size; ++j) {
          T computed = vIn[j];
          BOOST_TEST(detail::compare_type(expected[j], computed),
                     detail::type_to_string(computed)
                         << " at position " << j << " instead of "
                         << detail::type_to_string(expected[j])
                         << " for case: init multiplication in group " << i);
          if (!detail::compare_type(expected[j], computed))
            break;
        }
      }
    };

    test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(group_exclusive_scan, T, test_types) {
  const size_t elements_per_thread = 1;
  const auto   data_generator      = [](std::vector<T> &v, size_t local_size,
                                 size_t global_size) {
    for (size_t i = 0; i < global_size; ++i)
      v[i] =
          detail::initialize_type<T>(i) + detail::get_offset<T>(global_size, global_size);
  };

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      acc[global_linear_id] = sycl::exclusive_scan_over_group(g, local_value, std::plus<T>());
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig,size_t, size_t local_size,
                                        size_t global_size) {
      std::vector<T> expected(vOrig.size());

      for (size_t i = 0; i < global_size / local_size; ++i) {
        expected[i * local_size] = T{};
        for (size_t j = 1; j < local_size; ++j)
          expected[i * local_size + j] =
              expected[i * local_size + j - 1] + vOrig[i * local_size + j - 1];

        for (size_t j = i * local_size; j < (i + 1) * local_size; ++j) {
          T computed = vIn[j];
          BOOST_TEST(detail::compare_type(expected[j], computed),
                     detail::type_to_string(computed)
                         << " at position " << j << " instead of "
                         << detail::type_to_string(expected[j])
                         << " for case: no init in group " << i);

          if (!detail::compare_type(expected[j], computed))
            break;
        }
      }
    };

    test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);
  }

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      acc[global_linear_id] = sycl::exclusive_scan_over_group(
          g, local_value, detail::initialize_type<T>(10), std::plus<T>());
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig,size_t, size_t local_size,
                                        size_t global_size) {
      std::vector<T> expected(vOrig.size());

      for (size_t i = 0; i < global_size / local_size; ++i) {
        expected[i * local_size] = detail::initialize_type<T>(10);
        for (size_t j = 1; j < local_size; ++j)
          expected[i * local_size + j] =
              expected[i * local_size + j - 1] + vOrig[i * local_size + j - 1];

        for (size_t j = i * local_size; j < (i + 1) * local_size; ++j) {
          T computed = vIn[j];
          BOOST_TEST(detail::compare_type(expected[j], computed),
                     detail::type_to_string(computed)
                         << " at position " << j << " instead of "
                         << detail::type_to_string(expected[j])
                         << " for case: init in group " << i);
          if (!detail::compare_type(expected[j], computed))
            break;
        }
      }
    };

    test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(group_exclusive_scan_ptr, T, test_types) {
  const size_t elements_per_thread = 4;
  const auto   data_generator      = [](std::vector<T> &v, size_t local_size,
                                 size_t global_size) {
    for (size_t i = 0; i < global_size; ++i)
      v[i] = detail::initialize_type<T>(i) +
             detail::get_offset<T>(global_size, local_size * 2);
  };

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      auto local_size = g.get_local_range().size();
      auto start = acc.get_pointer() + (global_linear_id / local_size) * local_size * 2;
      auto end   = start + local_size * 2;
      auto out   = acc.get_pointer() + 2 * 4 * local_size +
                 (global_linear_id / local_size) * local_size * 2;

      sycl::joint_exclusive_scan(g, start.get(), end.get(), out.get(), std::plus<T>());
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig,size_t, size_t local_size,
                                        size_t global_size) {
      std::vector<T> expected(vOrig.size());

      for (size_t i = 0; i < global_size / local_size; ++i) {
        expected[i * 2 * local_size] = T{};
        for (size_t j = 1; j < local_size * 2; ++j)
          expected[i * 2 * local_size + j] =
              expected[i * 2 * local_size + j - 1] + vOrig[i * 2 * local_size + j - 1];

        for (size_t j = i * 2 * local_size; j < (i + 1) * local_size * 2; ++j) {
          T computed = vIn[j + global_size * 2];
          BOOST_TEST(detail::compare_type(expected[j], computed),
                     detail::type_to_string(computed)
                         << " at position " << j << " instead of "
                         << detail::type_to_string(expected[j])
                         << " for case: no init in group " << i);

          if (!detail::compare_type(expected[j], computed))
            break;
        }
      }
    };

    test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);
  }

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      auto local_size = g.get_local_range().size();
      auto start = acc.get_pointer() + (global_linear_id / local_size) * local_size * 2;
      auto end   = start + local_size * 2;
      auto out   = acc.get_pointer() + 2 * 4 * local_size +
                 (global_linear_id / local_size) * local_size * 2;

      sycl::joint_exclusive_scan(g, start.get(), end.get(), out.get(),
                                   detail::initialize_type<T>(10), std::plus<T>());
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig,size_t, size_t local_size,
                                        size_t global_size) {
      std::vector<T> expected(vOrig.size());

      for (size_t i = 0; i < global_size / local_size; ++i) {
        expected[i * 2 * local_size] = detail::initialize_type<T>(10);
        for (size_t j = 1; j < local_size * 2; ++j)
          expected[i * 2 * local_size + j] =
              expected[i * 2 * local_size + j - 1] + vOrig[i * 2 * local_size + j - 1];

        for (size_t j = i * 2 * local_size; j < (i + 1) * local_size * 2; ++j) {
          T computed = vIn[j + global_size * 2];
          BOOST_TEST_REQUIRE(detail::compare_type(expected[j], computed),
                     detail::type_to_string(computed)
                         << " at position " << j << " instead of "
                         << detail::type_to_string(expected[j])
                         << " for case: init in group " << i);
          if (!detail::compare_type(expected[j], computed))
            break;
        }
      }
    };

    test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(sub_group_exclusive_scan, T, test_types) {
  if(!sycl::queue{}.get_device().is_host()) {
    const size_t   elements_per_thread = 1;
    const auto     data_generator      = [](std::vector<T> &v, size_t local_size,
                                  size_t global_size) {
      for (size_t i = 0; i < global_size; ++i)
        v[i] =
            detail::initialize_type<T>(i) + detail::get_offset<T>(global_size, global_size);
    };

    {
      const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                      auto g, T local_value) {
        acc[global_linear_id] = sycl::exclusive_scan_over_group(sg, local_value, std::plus<T>());
      };
      const auto validation_function = [](const std::vector<T> &vIn,
                                          const std::vector<T> &vOrig,size_t subgroup_size, size_t local_size,
                                          size_t global_size) {
        std::vector<T> expected(vOrig.size());
        for (size_t i = 0; i < global_size / local_size; ++i) {
          expected[i * local_size] = T{};
          auto actual_warp_size    = local_size < subgroup_size ? local_size : subgroup_size;
          for (size_t j = 1; j < actual_warp_size; ++j)
            expected[i * local_size + j] =
                expected[i * local_size + j - 1] + vOrig[i * local_size + j - 1];

          for (size_t j = i * local_size; j < (i + 1) * actual_warp_size; ++j) {
            T computed = vIn[j];
            BOOST_TEST(detail::compare_type(expected[j], computed),
                      detail::type_to_string(computed)
                          << " at position " << j << " instead of "
                          << detail::type_to_string(expected[j]) << " for local_size "
                          << local_size << " and case: no init in group " << i);

            if (!detail::compare_type(expected[j], computed))
              break;
          }
        }
      };

      test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                            tested_function, validation_function);
    }

    {
      const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                      auto g, T local_value) {
        acc[global_linear_id] = sycl::exclusive_scan_over_group(
            sg, local_value, detail::initialize_type<T>(10), std::plus<T>());
      };
      const auto validation_function = [](const std::vector<T> &vIn,
                                          const std::vector<T> &vOrig,size_t subgroup_size, size_t local_size,
                                          size_t global_size) {
        std::vector<T> expected(vOrig.size());

        for (size_t i = 0; i < global_size / local_size; ++i) {
          expected[i * local_size] = detail::initialize_type<T>(10);
          auto actual_warp_size    = local_size < subgroup_size ? local_size : subgroup_size;
          for (size_t j = 1; j < actual_warp_size; ++j)
            expected[i * local_size + j] =
                expected[i * local_size + j - 1] + vOrig[i * local_size + j - 1];

          for (size_t j = i * local_size; j < (i + 1) * actual_warp_size; ++j) {
            T computed = vIn[j];
            BOOST_TEST(detail::compare_type(expected[j], computed),
                      detail::type_to_string(computed)
                          << " at position " << j << " instead of "
                          << detail::type_to_string(expected[j]) << " for local_size "
                          << local_size << " and case: init in group " << i);
            if (!detail::compare_type(expected[j], computed))
              break;
          }
        }
      };

      test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                            tested_function, validation_function);
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(group_inclusive_scan_mul, T, test_types) {
  const size_t elements_per_thread = 1;
  const auto   data_generator      = [](std::vector<T> &v, size_t local_size,
                                 size_t global_size) {
    for (size_t i = 0; i < global_size; ++i)
      v[i] = (i < 1) ? T{static_cast<T>(2)} : T{static_cast<T>(1)};
  };

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      acc[global_linear_id] =
          sycl::inclusive_scan_over_group(g, local_value, std::multiplies<T>());
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig,size_t, size_t local_size,
                                        size_t global_size) {
      std::vector<T> expected(vOrig.size());

      for (size_t i = 0; i < global_size / local_size; ++i) {
        expected[i * local_size] = vOrig[i * local_size];
        for (size_t j = 1; j < local_size; ++j)
          expected[i * local_size + j] =
              expected[i * local_size + j - 1] * vOrig[i * local_size + j];

        for (size_t j = i * local_size; j < (i + 1) * local_size; ++j) {
          T computed = vIn[j];
          BOOST_TEST(detail::compare_type(expected[j], computed),
                     detail::type_to_string(computed)
                         << " at position " << j << " instead of "
                         << detail::type_to_string(expected[j])
                         << " for case: no init multiplication in group " << i);
          if (!detail::compare_type(expected[j], computed))
            break;
        }
      }
    };

    test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(group_inclusive_scan, T, test_types) {
  const size_t elements_per_thread = 1;
  const auto   data_generator      = [](std::vector<T> &v, size_t local_size,
                                 size_t global_size) {
    for (size_t i = 0; i < global_size; ++i)
      v[i] =
          detail::initialize_type<T>(i) + detail::get_offset<T>(global_size, global_size);
  };

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      acc[global_linear_id] = sycl::inclusive_scan_over_group(g, local_value, std::plus<T>());
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig,size_t, size_t local_size,
                                        size_t global_size) {
      std::vector<T> expected(vOrig.size());

      for (size_t i = 0; i < global_size / local_size; ++i) {
        expected[i * local_size] = vOrig[i * local_size];
        for (size_t j = 1; j < local_size; ++j)
          expected[i * local_size + j] =
              expected[i * local_size + j - 1] + vOrig[i * local_size + j];

        for (size_t j = i * local_size; j < (i + 1) * local_size; ++j) {
          T computed = vIn[j];
          BOOST_TEST(detail::compare_type(expected[j], computed),
                     detail::type_to_string(computed)
                         << " at position " << j << " instead of "
                         << detail::type_to_string(expected[j])
                         << " for case: no init in group " << i);
          if (!detail::compare_type(expected[j], computed))
            break;
        }
      }
    };

    test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);
  }

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      acc[global_linear_id] = sycl::inclusive_scan_over_group(
          g, local_value, detail::initialize_type<T>(10), std::plus<T>());
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig,size_t, size_t local_size,
                                        size_t global_size) {
      std::vector<T> expected(vOrig.size());

      for (size_t i = 0; i < global_size / local_size; ++i) {
        expected[i * local_size] = vOrig[i * local_size] + detail::initialize_type<T>(10);
        for (size_t j = 1; j < local_size; ++j)
          expected[i * local_size + j] =
              expected[i * local_size + j - 1] + vOrig[i * local_size + j];

        for (size_t j = i * local_size; j < (i + 1) * local_size; ++j) {
          T computed = vIn[j];
          BOOST_TEST(detail::compare_type(expected[j], computed),
                     detail::type_to_string(computed)
                         << " at position " << j << " instead of "
                         << detail::type_to_string(expected[j])
                         << " for case: init in group " << i);
          if (!detail::compare_type(expected[j], computed))
            break;
        }
      }
    };

    test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(group_inclusive_scan_ptr, T, test_types) {
  const size_t elements_per_thread = 4;
  const auto   data_generator      = [](std::vector<T> &v, size_t local_size,
                                 size_t global_size) {
    for (size_t i = 0; i < global_size; ++i)
      v[i] = detail::initialize_type<T>(i) +
             detail::get_offset<T>(global_size, local_size * 2);
  };

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      auto local_size = g.get_local_range().size();
      auto start = acc.get_pointer() + (global_linear_id / local_size) * local_size * 2;
      auto end   = start + local_size * 2;
      auto out   = acc.get_pointer() + 2 * 4 * local_size +
                 (global_linear_id / local_size) * local_size * 2;

      sycl::joint_inclusive_scan(g, start.get(), end.get(), out.get(), std::plus<T>());
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig,size_t, size_t local_size,
                                        size_t global_size) {
      std::vector<T> expected(vOrig.size());

      for (size_t i = 0; i < global_size / local_size; ++i) {
        expected[i * 2 * local_size] = vOrig[i * 2 * local_size];
        for (size_t j = 1; j < local_size * 2; ++j)
          expected[i * 2 * local_size + j] =
              expected[i * 2 * local_size + j - 1] + vOrig[i * 2 * local_size + j];

        for (size_t j = i * 2 * local_size; j < (i + 1) * local_size * 2; ++j) {
          T computed = vIn[j + 2 * global_size];
          BOOST_TEST(detail::compare_type(expected[j], computed),
                     detail::type_to_string(computed)
                         << " at position " << j << " instead of "
                         << detail::type_to_string(expected[j]) << " for local_size "
                         << local_size << " and case: no init in group " << i);
          if (!detail::compare_type(expected[j], computed))
            break;
        }
      }
    };

    test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);
  }

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      auto local_size = g.get_local_range().size();
      auto start = acc.get_pointer() + (global_linear_id / local_size) * local_size * 2;
      auto end   = start + local_size * 2;
      auto out   = acc.get_pointer() + 2 * 4 * local_size +
                 (global_linear_id / local_size) * local_size * 2;

      sycl::joint_inclusive_scan(g, start.get(), end.get(), out.get(),
                                   std::plus<T>(), detail::initialize_type<T>(10));
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig,size_t, size_t local_size,
                                        size_t global_size) {
      std::vector<T> expected(vOrig.size());

      for (size_t i = 0; i < global_size / local_size; ++i) {
        expected[i * 2 * local_size] =
            vOrig[i * 2 * local_size] + detail::initialize_type<T>(10);
        for (size_t j = 1; j < local_size * 2; ++j)
          expected[i * 2 * local_size + j] =
              expected[i * 2 * local_size + j - 1] + vOrig[i * 2 * local_size + j];

        for (size_t j = i * 2 * local_size; j < (i + 1) * local_size * 2; ++j) {
          T computed = vIn[j + 2 * global_size];
          BOOST_TEST(detail::compare_type(expected[j], computed),
                     detail::type_to_string(computed)
                         << " at position " << j << " instead of "
                         << detail::type_to_string(expected[j]) << " for local_size "
                         << local_size << " and case: init in group " << i);
          if (!detail::compare_type(expected[j], computed))
            break;
        }
      }
    };

    test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);
  }
}


BOOST_AUTO_TEST_CASE_TEMPLATE(sub_group_inclusive_scan, T, test_types) {
  if(!sycl::queue{}.get_device().is_host()) {
    const size_t   elements_per_thread = 1;
    const auto     data_generator      = [](std::vector<T> &v, size_t local_size,
                                  size_t global_size) {
      for (size_t i = 0; i < global_size; ++i)
        v[i] =
            detail::initialize_type<T>(i) + detail::get_offset<T>(global_size, global_size);
    };

    {
      const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                      auto g, T local_value) {
        acc[global_linear_id] = sycl::inclusive_scan_over_group(sg, local_value, std::plus<T>());
      };
      const auto validation_function = [](const std::vector<T> &vIn,
                                          const std::vector<T> &vOrig, size_t subgroup_size, size_t local_size,
                                          size_t global_size) {
        std::vector<T> expected(vOrig.size());

        for (size_t i = 0; i < global_size / local_size; ++i) {
          expected[i * local_size] = vOrig[i * local_size];
          auto actual_warp_size    = local_size < subgroup_size ? local_size : subgroup_size;
          for (size_t j = 1; j < actual_warp_size; ++j)
            expected[i * local_size + j] =
                expected[i * local_size + j - 1] + vOrig[i * local_size + j];

          for (size_t j = i * local_size; j < (i + 1) * actual_warp_size; ++j) {
            T computed = vIn[j];
            BOOST_TEST(detail::compare_type(expected[j], computed),
                      detail::type_to_string(computed)
                          << " at position " << j << " instead of "
                          << detail::type_to_string(expected[j]) << " for local_size "
                          << local_size << " and case: no init in group " << i);
            if (!detail::compare_type(expected[j], computed))
              break;
          }
        }
      };

      test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                            tested_function, validation_function);
    }

    {
      const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                      auto g, T local_value) {
        acc[global_linear_id] = sycl::inclusive_scan_over_group(
            sg, local_value, detail::initialize_type<T>(10), std::plus<T>());
      };
      const auto validation_function = [](const std::vector<T> &vIn,
                                          const std::vector<T> &vOrig, size_t subgroup_size, size_t local_size,
                                          size_t global_size) {
        std::vector<T> expected(vOrig.size());

        for (size_t i = 0; i < global_size / local_size; ++i) {
          expected[i * local_size] = vOrig[i * local_size] + detail::initialize_type<T>(10);
          auto actual_warp_size    = local_size < subgroup_size ? local_size : subgroup_size;
          for (size_t j = 1; j < actual_warp_size; ++j)
            expected[i * local_size + j] =
                expected[i * local_size + j - 1] + vOrig[i * local_size + j];

          for (size_t j = i * local_size; j < (i + 1) * actual_warp_size; ++j) {
            T computed = vIn[j];
            BOOST_TEST(detail::compare_type(expected[j], computed),
                      detail::type_to_string(computed)
                          << " at position " << j << " instead of "
                          << detail::type_to_string(expected[j]) << " for local_size "
                          << local_size << " and case: init in group " << i);
            if (!detail::compare_type(expected[j], computed))
              break;
          }
        }
      };

      test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                            tested_function, validation_function);
    }
  }
}
BOOST_AUTO_TEST_SUITE_END()

#endif
