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

BOOST_AUTO_TEST_CASE(group_barrier) {
  using T = int;

  const size_t elements_per_thread = 1;
  const auto   data_generator      = [](std::vector<T> &v, size_t local_size,
                                 size_t global_size) {
    for (size_t i = 0; i < v.size(); ++i)
      v[i] = detail::initialize_type<T>(i);
  };

  {
    const auto tested_function = [=](auto acc, size_t global_linear_id,
                                     sycl::sub_group sg, auto g, T local_value) {
      int    tmp          = -10000;
      size_t local_id     = g.get_local_linear_id();
      auto   local_size   = g.get_local_range().size();
      size_t group_offset = (global_linear_id / local_size) * local_size;
      for (int i = 0; i < local_size; ++i) {
        if (local_id == i) {
          for (int j = 0; j < 10000; ++j)
            tmp++;
        }
        sycl::group_barrier(g);
        if (local_id == i)
          acc[group_offset + i] = tmp;
        sycl::group_barrier(g);
        tmp = acc[group_offset + i];
      }
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig, size_t subgroup_size, size_t local_size,
                                        size_t global_size) {
      for (size_t i = 0; i < vIn.size(); ++i) {
        T expected = (i % local_size) * 10000;
        T computed = vIn[i];

        BOOST_TEST(detail::compare_type(expected, computed),
                   detail::type_to_string(computed)
                       << " at position " << i << " instead of "
                       << detail::type_to_string(expected)
                       << " for group: " << i / local_size);

        if (!detail::compare_type(expected, computed))
          break;
      }
    };

    test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(group_broadcast, T, test_types) {
  const size_t elements_per_thread = 1;
  const auto   data_generator      = [](std::vector<T> &v, size_t local_size,
                                 size_t global_size) {
    for (size_t i = 0; i < v.size(); ++i)
      v[i] = detail::initialize_type<T>(i) + detail::get_offset<T>(global_size);
  };

  {
    const auto tested_function = [=](auto acc, size_t global_linear_id,
                                     sycl::sub_group sg, auto g, T local_value) {
      acc[global_linear_id] = sycl::group_broadcast(g, local_value);
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig,size_t subgroup_size, size_t local_size,
                                        size_t global_size) {
      for (size_t i = 0; i < vIn.size(); ++i) {
        T expected = detail::initialize_type<T>(((int)i / local_size) * local_size) +
                     detail::get_offset<T>(global_size, 1);
        T computed = vIn[i];

        BOOST_TEST(detail::compare_type(expected, computed),
                   detail::type_to_string(computed)
                       << " at position " << i << " instead of "
                       << detail::type_to_string(expected) << " for case: no id");

        if (!detail::compare_type(expected, computed))
          break;
      }
    };

    test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);
  }

  {
    const auto tested_function = [=](auto acc, size_t global_linear_id,
                                     sycl::sub_group sg, auto g, T local_value) {
      acc[global_linear_id] = sycl::group_broadcast(g, local_value, 10);
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig,size_t subgroup_size, size_t local_size,
                                        size_t global_size) {
      for (size_t i = 0; i < vIn.size(); ++i) {
        T expected = detail::initialize_type<T>(((int)i / local_size) * local_size + 10) +
                     detail::get_offset<T>(global_size, 1);
        T computed = vIn[i];

        BOOST_TEST(detail::compare_type(expected, computed),
                   detail::type_to_string(computed)
                       << " at position " << i << " instead of "
                       << detail::type_to_string(expected) << " for case: linear id");

        if (!detail::compare_type(expected, computed))
          break;
      }
    };

    test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);
  }

  {
    const auto tested_function_1d = [](auto acc, size_t global_linear_id,
                                       sycl::sub_group sg, auto g, T local_value) {
      acc[global_linear_id] = sycl::group_broadcast(g, local_value, sycl::id<1>(10));
    };
    const auto tested_function_2d = [](auto acc, size_t global_linear_id,
                                       sycl::sub_group sg, auto g, T local_value) {
      acc[global_linear_id] = sycl::group_broadcast(g, local_value, sycl::id<2>(0, 10));
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig,size_t subgroup_size, size_t local_size,
                                        size_t global_size) {
      for (size_t i = 0; i < vIn.size(); ++i) {
        T expected = detail::initialize_type<T>(((int)i / local_size) * local_size + 10) +
                     detail::get_offset<T>(global_size, 1);
        T computed = vIn[i];

        BOOST_TEST(detail::compare_type(expected, computed),
                   detail::type_to_string(computed)
                       << " at position " << i << " instead of "
                       << detail::type_to_string(expected) << " for case: id");

        if (!detail::compare_type(expected, computed))
          break;
      }
    };

    test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function_1d, validation_function);

    test_nd_group_function_2d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function_2d, validation_function);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(sub_group_broadcast, T, test_types) {
  if(!sycl::queue{}.get_device().is_host()) {
    const size_t   elements_per_thread = 1;

    const auto data_generator = [](std::vector<T> &v, size_t local_size,
                                  size_t global_size) {
      for (size_t i = 0; i < v.size(); ++i)
        v[i] = detail::initialize_type<T>(i) + detail::get_offset<T>(global_size);
    };

    {
      const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                      auto g, T local_value) {
        acc[global_linear_id] = sycl::group_broadcast(sg, local_value);
      };
      const auto validation_function = [](const std::vector<T> &vIn,
                                          const std::vector<T> &vOrig, size_t subgroup_size, size_t local_size,
                                          size_t global_size) {
        for (size_t i = 0; i < vIn.size(); ++i) {
          int expected_base = i % local_size;
          expected_base = ((int)expected_base / subgroup_size) *
                          subgroup_size;
          expected_base += ((int)i / local_size) * local_size;

          T expected = detail::initialize_type<T>(expected_base) +
                      detail::get_offset<T>(global_size);
          T computed = vIn[i];

          BOOST_TEST(detail::compare_type(expected, computed),
                    detail::type_to_string(computed)
                        << " at position " << i << " instead of "
                        << detail::type_to_string(expected) << " for case: no id");

          if (!detail::compare_type(expected, computed))
            break;
        }
      };

      test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                            tested_function, validation_function);
    }

    {
      const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                      auto g, T local_value) {
        acc[global_linear_id] = sycl::group_broadcast(sg, local_value, 7);
      };
      const auto validation_function = [](const std::vector<T> &vIn,
                                          const std::vector<T> &vOrig, size_t subgroup_size, size_t local_size,
                                          size_t global_size) {
        for (size_t i = 0; i < vIn.size(); ++i) {
          int expected_base = i % local_size;
          expected_base     = ((int)expected_base / subgroup_size) * subgroup_size;
          expected_base += ((int)i / local_size) * local_size + 7;

          T expected = detail::initialize_type<T>(expected_base) +
                      detail::get_offset<T>(global_size);
          T computed = vIn[i];

          BOOST_TEST(detail::compare_type(expected, computed),
                    detail::type_to_string(computed)
                        << " at position " << i << " instead of "
                        << detail::type_to_string(expected) << " for case: linear id");

          if (!detail::compare_type(expected, computed))
            break;
        }
      };

      test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                            tested_function, validation_function);
    }

    {
      const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                      auto g, T local_value) {
        acc[global_linear_id] = sycl::group_broadcast(sg, local_value, sycl::id<1>(7));
      };
      const auto validation_function = [](const std::vector<T> &vIn,
                                          const std::vector<T> &vOrig,size_t subgroup_size, size_t local_size,
                                          size_t global_size) {
        for (size_t i = 0; i < vIn.size(); ++i) {
          int expected_base = i % local_size;
          expected_base     = ((int)expected_base / subgroup_size) * subgroup_size;
          expected_base += ((int)i / local_size) * local_size + 7;

          T expected = detail::initialize_type<T>(expected_base) +
                      detail::get_offset<T>(global_size);
          T computed = vIn[i];

          BOOST_TEST(detail::compare_type(expected, computed),
                    detail::type_to_string(computed)
                        << " at position " << i << " instead of "
                        << detail::type_to_string(expected) << " for case: id");

          if (!detail::compare_type(expected, computed))
            break;
        }
      };

      test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                            tested_function, validation_function);
    }
  }
}

#if !defined(REDUCED_LOCAL_MEM_USAGE)
BOOST_AUTO_TEST_CASE_TEMPLATE(group_shuffle_like, T, test_types) {
  const size_t elements_per_thread = 1;
  const auto   data_generator      = [](std::vector<T> &v, size_t local_size,
                                 size_t global_size) {
    for (size_t i = 0; i < v.size(); ++i)
      v[i] = detail::initialize_type<T>(i) + detail::get_offset<T>(global_size);
  };
  {
    const auto tested_function = [=](auto acc, size_t global_linear_id,
                                     sycl::sub_group sg, auto g, T local_value) {
      acc[global_linear_id] = sycl::shift_group_left(g, local_value, 1);
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig,size_t, size_t local_size,
                                        size_t global_size) {
      for (size_t i = 0; i < vIn.size(); ++i) {
        T expected =
            detail::initialize_type<T>(i + 1) + detail::get_offset<T>(global_size, 1);

        // output only defined if target is in group
        if (static_cast<int>((i + 1) / local_size) + 1 > 1)
          continue;

        T computed = vIn[i];

        BOOST_TEST(detail::compare_type(expected, computed),
                   detail::type_to_string(computed)
                       << " at position " << i << " instead of "
                       << detail::type_to_string(expected)
                       << " for case: group, shift left");

        if (!detail::compare_type(expected, computed))
          break;
      }
    };

    test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);
  }

  {
    const auto tested_function = [=](auto acc, size_t global_linear_id,
                                     sycl::sub_group sg, auto g, T local_value) {
      acc[global_linear_id] = sycl::shift_group_right(g, local_value, 1);
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig,size_t, size_t local_size,
                                        size_t global_size) {
      for (size_t i = 0; i < vIn.size(); ++i) {
        T expected =
            detail::initialize_type<T>(i - 1) + detail::get_offset<T>(global_size, 1);

        // output only defined if target is in group
        if (i % local_size == 0)
          continue;

        T computed = vIn[i];

        BOOST_TEST(detail::compare_type(expected, computed),
                   detail::type_to_string(computed)
                       << " at position " << i << " instead of "
                       << detail::type_to_string(expected)
                       << " for case: group, shift right");

        if (!detail::compare_type(expected, computed))
          break;
      }
    };

    test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);
  }

  {
    const auto tested_function = [=](auto acc, size_t global_linear_id,
                                     sycl::sub_group sg, auto g, T local_value) {
      acc[global_linear_id] = sycl::permute_group_by_xor(g, local_value, 1);
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig,size_t, size_t local_size,
                                        size_t global_size) {
      for (size_t i = 0; i < vIn.size(); ++i) {
        T expected =
            detail::initialize_type<T>(i ^ 1) + detail::get_offset<T>(global_size, 1);

        // output only defined if target is in group
        if (static_cast<int>((i + 1) / local_size) + 1 > 1)
          continue;

        T computed = vIn[i];

        BOOST_TEST(detail::compare_type(expected, computed),
                   detail::type_to_string(computed)
                       << " at position " << i << " instead of "
                       << detail::type_to_string(expected)
                       << " for case: group, permute xor");

        if (!detail::compare_type(expected, computed))
          break;
      }
    };

    test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);
  }

  {
    const auto tested_function = [=](auto acc, size_t global_linear_id,
                                     sycl::sub_group sg, auto g, T local_value) {
      acc[global_linear_id] =
          sycl::select_from_group(g, local_value, sycl::id<g.dimensions>());
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig, size_t, size_t local_size,
                                        size_t global_size) {
      for (size_t i = 0; i < vIn.size(); ++i) {
        T expected =
            detail::initialize_type<T>(static_cast<int>(i / local_size) * local_size) +
            detail::get_offset<T>(global_size, 1);

        T computed = vIn[i];

        BOOST_TEST(detail::compare_type(expected, computed),
                   detail::type_to_string(computed)
                       << " at position " << i << " instead of "
                       << detail::type_to_string(expected) << " for case: group, select");

        if (!detail::compare_type(expected, computed))
          break;
      }
    };

    test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);
  }
}
#endif
BOOST_AUTO_TEST_CASE_TEMPLATE(subgroup_shuffle_like, T, test_types) {
  if(!sycl::queue{}.get_device().is_host()) {
    const size_t elements_per_thread = 1;
    const auto   data_generator      = [](std::vector<T> &v, size_t local_size,
                                  size_t global_size) {
      for (size_t i = 0; i < v.size(); ++i)
        v[i] = detail::initialize_type<T>(i) + detail::get_offset<T>(global_size);
    };

    {
      const auto tested_function = [=](auto acc, size_t global_linear_id,
                                      sycl::sub_group sg, auto g, T local_value) {
        acc[global_linear_id] = sycl::shift_group_left(sg, local_value, 1);
      };
      const auto validation_function = [](const std::vector<T> &vIn,
                                          const std::vector<T> &vOrig,size_t subgroup_size, size_t local_size,
                                          size_t global_size) {
        for (size_t i = 0; i < global_size / local_size; ++i) {
          for (size_t j = 0; j < (local_size + subgroup_size - 1) / subgroup_size; ++j) {
            for (size_t k = 0; k < subgroup_size; ++k) {
              size_t local_index  = j * subgroup_size + k;
              size_t global_index = i * local_size + local_index;

              if (local_index >= local_size) // keep to work group size
                break;

              T expected = detail::initialize_type<T>(global_index + 1) +
                          detail::get_offset<T>(global_size, 1);

              if (local_index == local_size - 1 ||
                  k == subgroup_size - 1) // not defined for last work item
                continue;

              T computed = vIn[global_index];

              BOOST_TEST(detail::compare_type(expected, computed),
                        detail::type_to_string(computed)
                            << " at position " << global_index << " instead of "
                            << detail::type_to_string(expected)
                            << " for case: sub_group, shift left, local size: "
                            << local_size);

              if (!detail::compare_type(expected, computed))
                return;
            }
          }
        }
      };

      test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                            tested_function, validation_function);
    }

    {
      const auto tested_function = [=](auto acc, size_t global_linear_id,
                                      sycl::sub_group sg, auto g, T local_value) {
        acc[global_linear_id] = sycl::shift_group_right(sg, local_value, 1);
      };
      const auto validation_function = [](const std::vector<T> &vIn,
                                          const std::vector<T> &vOrig,size_t subgroup_size, size_t local_size,
                                          size_t global_size) {
        for (size_t i = 0; i < global_size / local_size; ++i) {
          for (size_t j = 0; j < (local_size + subgroup_size - 1) / subgroup_size; ++j) {
            for (size_t k = 0; k < subgroup_size; ++k) {
              size_t local_index  = j * subgroup_size + k;
              size_t global_index = i * local_size + local_index;

              if (local_index >= local_size) // keep to work group size
                break;

              T expected = detail::initialize_type<T>(global_index - 1) +
                          detail::get_offset<T>(global_size, 1);

              if (k == 0) // not defined for first work item
                continue;

              T computed = vIn[global_index];

              BOOST_TEST(detail::compare_type(expected, computed),
                        detail::type_to_string(computed)
                            << " at position " << global_index << " instead of "
                            << detail::type_to_string(expected)
                            << " for case: sub_group, shift right, local size: "
                            << local_size);

              if (!detail::compare_type(expected, computed))
                return;
            }
          }
        }
      };
      test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                            tested_function, validation_function);
    }

    {
      const auto tested_function = [=](auto acc, size_t global_linear_id,
                                      sycl::sub_group sg, auto g, T local_value) {
        acc[global_linear_id] = sycl::permute_group_by_xor(sg, local_value, 1);
      };
      const auto validation_function = [](const std::vector<T> &vIn,
                                          const std::vector<T> &vOrig,size_t subgroup_size, size_t local_size,
                                          size_t global_size) {
        for (size_t i = 0; i < global_size / local_size; ++i) {
          for (size_t j = 0; j < (local_size + subgroup_size - 1) / subgroup_size; ++j) {
            for (size_t k = 0; k < subgroup_size; ++k) {
              size_t local_index  = j * subgroup_size + k;
              size_t global_index = i * local_size + local_index;

              if (local_index >= local_size) // keep to work group size
                break;

              T expected = detail::initialize_type<T>(i*local_size + (local_index ^ 1)) +
                          detail::get_offset<T>(global_size, 1);

              if ((local_index ^ 1) >= local_size ||
                  (k ^ 1) >= subgroup_size) // only defined if target is in subgroup
                continue;

              T computed = vIn[global_index];

              BOOST_TEST(detail::compare_type(expected, computed),
                        detail::type_to_string(computed)
                            << " at position " << global_index << " instead of "
                            << detail::type_to_string(expected)
                            << " for case: sub_group, permute xor, local size: "
                            << local_size);

              if (!detail::compare_type(expected, computed))
                break;
            }
          }
        }
      };

      test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                            tested_function, validation_function);
    }
    
    {
      const auto tested_function = [=](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                      auto g, T local_value) {
        acc[global_linear_id] = sycl::select_from_group(sg, local_value, sycl::id<1>());
      };
      const auto validation_function = [](const std::vector<T> &vIn,
                                          const std::vector<T> &vOrig, size_t subgroup_size, size_t local_size,
                                          size_t global_size) {
          for (size_t i = 0; i < global_size / local_size; ++i) {
            for (size_t j = 0; j < (local_size + subgroup_size - 1) / subgroup_size; ++j) {
              for (size_t k = 0; k < subgroup_size; ++k) {
                size_t local_index  = j * subgroup_size + k;
                size_t global_index = i * local_size + local_index;

                if (local_index >= local_size) // keep to work group size
                  break;

                T expected = detail::initialize_type<T>(i*local_size + j*subgroup_size) +
                            detail::get_offset<T>(global_size, 1);

                T computed = vIn[global_index];

                BOOST_TEST(detail::compare_type(expected, computed),
                          detail::type_to_string(computed)
                              << " at position " << global_index << " instead of "
                              << detail::type_to_string(expected)
                              << " for case: sub_group, select, local size: "
                              << local_size);

                if (!detail::compare_type(expected, computed))
                  return;
              }
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
