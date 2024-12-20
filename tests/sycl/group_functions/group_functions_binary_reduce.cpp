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

#include <cstddef>

#include "../sycl_test_suite.hpp"
#include "group_functions.hpp"

#ifdef HIPSYCL_ENABLE_GROUP_ALGORITHM_TESTS

using namespace cl;

BOOST_FIXTURE_TEST_SUITE(group_functions_tests, reset_device_fixture)

BOOST_AUTO_TEST_CASE(group_x_of_local) {
  using T = char;

  const size_t elements_per_thread = 1;
  const auto   data_generator      = [](std::vector<T> &v, size_t local_size,
                                 size_t global_size) {
    detail::create_bool_test_data(v, local_size, global_size);
  };

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      acc[global_linear_id] = sycl::any_of_group(g, static_cast<bool>(local_value));
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig, size_t subgroup_size, size_t local_size,
                                        size_t global_size) {
      detail::check_binary_reduce<T, __LINE__>(vIn,vOrig, local_size, global_size,
                                               std::vector<bool>{true, false, true, true},
                                               "any_of");
    };

    test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);
  }

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      acc[global_linear_id] = sycl::all_of_group(g, static_cast<bool>(local_value));
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig, size_t subgroup_size, size_t local_size,
                                        size_t global_size) {
      detail::check_binary_reduce<T, __LINE__>(
          vIn, vOrig, local_size, global_size, std::vector<bool>{false, false, false, true},
          "all_of");
    };

    test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);
  }

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      acc[global_linear_id] = sycl::none_of_group(g, static_cast<bool>(local_value));
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig, size_t subgroup_size, size_t local_size,
                                        size_t global_size) {
      detail::check_binary_reduce<T, __LINE__>(
          vIn,vOrig, local_size, global_size, std::vector<bool>{false, true, false, false},
          "none_of");
    };

    test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);
  }
}

BOOST_AUTO_TEST_CASE(sub_group_x_of_local) {
  if(!sycl::queue{}.get_device().is_host()) {
    using T = char;

    const size_t   elements_per_thread = 1;

    const auto data_generator = [](std::vector<T> &v, size_t local_size,
                                  size_t global_size) {
      detail::create_bool_test_data(v, local_size, global_size);
    };

    {
      const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                      auto g, T local_value) {
        acc[global_linear_id] = sycl::any_of_group(sg, static_cast<bool>(local_value));
      };
      const auto validation_function = [=](const std::vector<T> &vIn,
                                          const std::vector<T> &vOrig, size_t subgroup_size, size_t local_size,
                                          size_t global_size) {
        detail::check_binary_reduce<T, __LINE__>(vIn,vOrig, local_size, global_size,
                                                std::vector<bool>{true, false, true, true},
                                                "any_of", subgroup_size);
      };

      test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                            tested_function, validation_function);
    }

    {
      const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                      auto g, T local_value) {
        acc[global_linear_id] = sycl::all_of_group(sg, static_cast<bool>(local_value));
      };
      const auto validation_function = [=](const std::vector<T> &vIn,
                                          const std::vector<T> &vOrig, size_t subgroup_size, size_t local_size,
                                          size_t global_size) {
        detail::check_binary_reduce<T, __LINE__>(
            vIn,vOrig, local_size, global_size, std::vector<bool>{false, false, false, true},
            "all_of bool sub group", subgroup_size);
      };

      test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                            tested_function, validation_function);
    }

    {
      const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                      auto g, T local_value) {
        acc[global_linear_id] = sycl::none_of_group(sg, static_cast<bool>(local_value));
      };
      const auto validation_function = [=](const std::vector<T> &vIn,
                                          const std::vector<T> &vOrig, size_t subgroup_size, size_t local_size,
                                          size_t global_size) {
        detail::check_binary_reduce<T, __LINE__>(
            vIn,vOrig, local_size, global_size, std::vector<bool>{false, true, false, false},
            "none_of", subgroup_size);
      };

      test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                            tested_function, validation_function);
    }
  }
}

BOOST_AUTO_TEST_CASE(group_x_of_ptr_function) {
  using T = char;

  const size_t elements_per_thread = 3;
  const auto   data_generator      = [](std::vector<T> &v, size_t local_size,
                                 size_t global_size) {
    detail::create_bool_test_data(v, local_size * 2, global_size * 2);
  };

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      auto local_size  = g.get_local_range().size();
      auto global_size = 4 * local_size;
      auto start = acc.get_pointer() + (global_linear_id / local_size) * local_size * 2;
      auto end   = start + local_size * 2;

      auto local = sycl::joint_any_of(g, start.get(), end.get(), std::logical_not<T>());
      sycl::group_barrier(g);
      acc[global_linear_id + 2 * global_size] = local;
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig, size_t subgroup_size, size_t local_size,
                                        size_t global_size) {
      detail::check_binary_reduce<T, __LINE__>(vIn,vOrig, local_size, global_size,
                                               std::vector<bool>{true, true, true, false},
                                               "any_of", 0, 2 * global_size);
    };

    test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);
  }

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      auto local_size  = g.get_local_range().size();
      auto global_size = 4 * local_size;
      auto start = acc.get_pointer() + (global_linear_id / local_size) * local_size * 2;
      auto end   = start + local_size * 2;

      auto local = sycl::joint_all_of(g, start.get(), end.get(), std::logical_not<T>());
      sycl::group_barrier(g);
      acc[global_linear_id + 2 * global_size] = local;
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig, size_t subgroup_size, size_t local_size,
                                        size_t global_size) {
      detail::check_binary_reduce<T, __LINE__>(
          vIn,vOrig, local_size, global_size, std::vector<bool>{false, true, false, false},
          "all_of", 0, 2 * global_size);
    };

    test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);
  }

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      auto local_size  = g.get_local_range().size();
      auto global_size = 4 * local_size;
      auto start = acc.get_pointer() + (global_linear_id / local_size) * local_size * 2;
      auto end   = start + local_size * 2;

      auto local =
          sycl::joint_none_of(g, start.get(), end.get(), std::logical_not<T>());
      sycl::group_barrier(g);
      acc[global_linear_id + 2 * global_size] = local;
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig, size_t subgroup_size, size_t local_size,
                                        size_t global_size) {
      detail::check_binary_reduce<T, __LINE__>(
          vIn,vOrig, local_size, global_size, std::vector<bool>{false, false, false, true},
          "none_of", 0, 2 * global_size);
    };

    test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);
  }
}

BOOST_AUTO_TEST_CASE(group_x_of_function) {
  using T = char;

  const size_t elements_per_thread = 1;
  const auto   data_generator      = [](std::vector<T> &v, size_t local_size,
                                 size_t global_size) {
    detail::create_bool_test_data(v, local_size, global_size);
  };

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      acc[global_linear_id] =
          sycl::any_of_group(g, static_cast<bool>(local_value), std::logical_not<T>());
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig, size_t subgroup_size, size_t local_size,
                                        size_t global_size) {
      detail::check_binary_reduce<T, __LINE__>(vIn,vOrig, local_size, global_size,
                                               std::vector<bool>{true, true, true, false},
                                               "any_of");
    };

    test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);
  }

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      acc[global_linear_id] =
          sycl::all_of_group(g, static_cast<bool>(local_value), std::logical_not<T>());
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig, size_t subgroup_size, size_t local_size,
                                        size_t global_size) {
      detail::check_binary_reduce<T, __LINE__>(
          vIn,vOrig, local_size, global_size, std::vector<bool>{false, true, false, false},
          "all_of");
    };

    test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);
  }

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      acc[global_linear_id] =
          sycl::none_of_group(g, static_cast<bool>(local_value), std::logical_not<T>());
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig, size_t subgroup_size, size_t local_size,
                                        size_t global_size) {
      detail::check_binary_reduce<T, __LINE__>(
          vIn,vOrig, local_size, global_size, std::vector<bool>{false, false, false, true},
          "none_of");
    };

    test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);
  }
}

BOOST_AUTO_TEST_CASE(sub_group_x_of_function) {
  if(!sycl::queue{}.get_device().is_host()) {
    using T = char;

    const size_t   elements_per_thread = 1;

    const auto data_generator = [](std::vector<T> &v, size_t local_size,
                                  size_t global_size) {
      detail::create_bool_test_data(v, local_size, global_size);
    };

    {
      const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                      auto g, T local_value) {
        acc[global_linear_id] =
            sycl::any_of_group(sg, static_cast<bool>(local_value), std::logical_not<T>());
      };
      const auto validation_function = [=](const std::vector<T> &vIn,
                                          const std::vector<T> &vOrig, size_t subgroup_size, size_t local_size,
                                          size_t global_size) {
        detail::check_binary_reduce<T, __LINE__>(vIn,vOrig, local_size, global_size,
                                                std::vector<bool>{true, true, true, false},
                                                "any_of", subgroup_size);
      };

      test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                            tested_function, validation_function);
    }

    {
      const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                      auto g, T local_value) {
        acc[global_linear_id] =
            sycl::all_of_group(sg, static_cast<bool>(local_value), std::logical_not<T>());
      };
      const auto validation_function = [=](const std::vector<T> &vIn,
                                          const std::vector<T> &vOrig, size_t subgroup_size, size_t local_size,
                                          size_t global_size) {
        detail::check_binary_reduce<T, __LINE__>(
            vIn,vOrig, local_size, global_size, std::vector<bool>{false, true, false, false},
            "all_of", subgroup_size);
      };

      test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                            tested_function, validation_function);
    }

    {
      const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                      auto g, T local_value) {
        acc[global_linear_id] =
            sycl::none_of_group(sg, static_cast<bool>(local_value), std::logical_not<T>());
      };
      const auto validation_function = [=](const std::vector<T> &vIn,
                                          const std::vector<T> &vOrig, size_t subgroup_size, size_t local_size,
                                          size_t global_size) {
        detail::check_binary_reduce<T, __LINE__>(
            vIn,vOrig, local_size, global_size, std::vector<bool>{false, false, false, true},
            "none_of", subgroup_size);
      };

      test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                            tested_function, validation_function);
    }
  }
}
BOOST_AUTO_TEST_SUITE_END()

#endif