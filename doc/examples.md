The following code adds two vectors:
```cpp
#include <cassert>
#include <iostream>

#include <sycl/sycl.hpp>

using data_type = float;

std::vector<data_type> add(sycl::queue& q,
                           const std::vector<data_type>& a,
                           const std::vector<data_type>& b) {
  std::vector<data_type> c(a.size());

  assert(a.size() == b.size());

  data_type* dev_a = sycl::malloc_device<data_type>(a.size(), q);
  data_type* dev_b = sycl::malloc_device<data_type>(a.size(), q);
  data_type* dev_c = sycl::malloc_device<data_type>(a.size(), q);

  q.memcpy(dev_a, a.data(), sizeof(T) * a.size());
  q.memcpy(dev_b, b.data(), sizeof(T) * b.size());
  q.memcpy(dev_c, c.data(), sizeof(T) * c.size());

  q.parallel_for(a.size(), [=](sycl::id<1> idx){
    dev_c[idx] = dev_a[idx] + dev_b[idx];
  });

  q.memcpy(c.data(), dev_c, sizeof(T) * c.size());
  q.wait();

  sycl::free(dev_a, q);
  sycl::free(dev_b, q);
  sycl::free(dev_c, q);

  return c;
}

int main()
{
  sycl::queue q{sycl::property::queue::in_order{}};
  std::vector<data_type> a = {1.f, 2.f, 3.f, 4.f, 5.f};
  std::vector<data_type> b = {-1.f, 2.f, -3.f, 4.f, -5.f};
  auto result = add(q, a, b);

  std::cout << "Result: " << std::endl;
  for(const auto x: result)
    std::cout << x << std::endl;
}

```
