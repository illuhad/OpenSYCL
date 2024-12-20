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
#include "hipSYCL/common/dylib_loader.hpp"

#include <cassert>
#include <string>

#ifndef _WIN32
#include <dlfcn.h>
#else
#include <windows.h>

// Adapted from: https://stackoverflow.com/a/17387176
// Returns the last Win32 error, in string format. Returns an empty string if
// there is no error.
static std::string format_win32_error(DWORD errorMessageID) {
  if (errorMessageID == 0) {
    return std::string(); // No error message has been recorded
  }

  LPSTR messageBuffer = nullptr;

  // Ask Win32 to give us the string version of that message ID.
  // The parameters we pass in, tell Win32 to create the buffer that holds the
  // message for us (because we don't yet know how long the message string will
  // be).
  const std::size_t size = FormatMessageA(
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
          FORMAT_MESSAGE_IGNORE_INSERTS,
      NULL, errorMessageID, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
      (LPSTR)&messageBuffer, 0, NULL);

  // Copy the error message into a std::string.
  std::string message(messageBuffer, size);
  if (message[message.size() - 1] == '\n')
    message[message.size() - 1] = '\0';
  if (message[message.size() - 2] == '\r')
    message[message.size() - 2] = '\0';

  // Free the Win32's string's buffer.
  LocalFree(messageBuffer);

  return message;
}
#endif

namespace hipsycl::common {

void close_library(void *handle, std::string &message) {
#ifndef _WIN32
  if (dlclose(handle)) {
    message = "dlclose() failed";
  }
#else
  if (!FreeLibrary(static_cast<HMODULE>(handle))) {
    message = "FreeLibrary() failed";
  }
#endif
}

void *load_library(const std::string &filename, std::string &message) {
#ifndef _WIN32
  if (void *handle = dlopen(filename.c_str(), RTLD_NOW)) {
    assert(handle != nullptr);
    return handle;
  } else {
    message = "Could not load library: " + filename;
    if (const char *err = dlerror()) {
      message += std::string(" (") + err + ")";
    }
  }
#else
  if (HMODULE handle = LoadLibraryExA(filename.c_str(), nullptr,
                                      LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)) {
    return static_cast<void *>(handle);
  } else if (HMODULE handle = LoadLibraryA(filename.c_str())) {
    return static_cast<void *>(handle);
  } else {
    DWORD errorCode = GetLastError();
    message = "Could not load library: " + filename +
              " with: " + format_win32_error(errorCode) + " (" +
              std::to_string(errorCode) + ")";
  }
#endif
  return nullptr;
}

void *get_symbol_from_library(void *handle, const std::string &symbolName,
                              std::string &message) {
#ifndef _WIN32
  void *symbol = dlsym(handle, symbolName.c_str());
  if (const char *err = dlerror()) {
    message = "Could not find symbol name: " + symbolName;
    message += std::string(" (") + err + ")";
  } else {
    return symbol;
  }
#else
  if (FARPROC symbol =
          GetProcAddress(static_cast<HMODULE>(handle), symbolName.c_str())) {
    return reinterpret_cast<void *>(symbol);
  } else {
    DWORD errorCode = GetLastError();
    message = std::string(loader) +
              ": Could not find symbol name: " + symbolName +
              " with: " + format_win32_error(errorCode) + " (" +
              std::to_string(errorCode) + ")";
  }
#endif
  return nullptr;
}
} // namespace hipsycl::common
