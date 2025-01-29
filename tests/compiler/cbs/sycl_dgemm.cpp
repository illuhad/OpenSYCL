// REQUIRES: cbs
// RUN: %acpp %s -o %t --acpp-targets=omp --acpp-use-accelerated-cpu
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=omp --acpp-use-accelerated-cpu -O
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: ACPP_VISIBILITY_MASK=omp; %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O
// RUN: ACPP_VISIBILITY_MASK=omp; %t | FileCheck %s

// adapted from https://github.com/UoB-HPC/sycl_dgemm/blob/main/dgemm.cpp

#include <CL/sycl.hpp>
using namespace cl;

const double Aval = 2.0;
const double Bval = 0.50;

void matmul_blocked(sycl::queue &Q, const size_t Ndim, const size_t Mdim, const size_t Pdim, sycl::buffer<double, 2> &A, sycl::buffer<double, 2> &B, sycl::buffer<double, 2> &C)
{

  const size_t Bsize = 16;
  assert(Ndim % Bsize == 0);
  assert(Mdim % Bsize == 0);
  assert(Pdim % Bsize == 0);

  Q.submit([&](sycl::handler &cgh) {
     // sycl::accessor a {A, cgh, sycl::read_only};
     // sycl::accessor b {B, cgh, sycl::read_only};
     // sycl::accessor c {C, cgh, sycl::read_write};
     auto a = A.get_access<sycl::access_mode::read>(cgh);
     auto b = B.get_access<sycl::access_mode::read>(cgh);
     auto c = C.get_access<sycl::access_mode::read_write>(cgh);

     sycl::accessor<double, 2, sycl::access_mode::read_write, sycl::access::target::local> Awrk({Bsize, Bsize}, cgh);
     sycl::accessor<double, 2, sycl::access_mode::read_write, sycl::access::target::local> Bwrk({Bsize, Bsize}, cgh);

     cgh.parallel_for(sycl::nd_range<2>{{Ndim, Mdim}, {Bsize, Bsize}}, [=](sycl::nd_item<2> idx) {
       // This work-item will compute C(i,j)
       const size_t i = idx.get_global_id(0);
       const size_t j = idx.get_global_id(1);

       // Element C(i,j) is in block C(Iblk, Jblk)
       const size_t Iblk = idx.get_group(0);
       const size_t Jblk = idx.get_group(1);

       // C(i,j) is element C(iloc, jloc) of block C(Iblk, Jblk)
       const size_t iloc = idx.get_local_id(0);
       const size_t jloc = idx.get_local_id(1);

       // Number of blocks
       const size_t Nblk = Ndim / Bsize;
       const size_t Mblk = Mdim / Bsize;
       const size_t Pblk = Pdim / Bsize;

       double accumulator{0};
       for(int Kblk = 0; Kblk < Pblk; ++Kblk)
       {

         // Copy A and B into local memory
         Awrk[iloc][jloc] = a[Iblk * Bsize + iloc][Kblk * Bsize + jloc];
         Bwrk[iloc][jloc] = b[Kblk * Bsize + iloc][Jblk * Bsize + jloc];
         // sycl::group_barrier(idx.get_group());
         idx.barrier();

         // Compute matmul for block
         for(int kloc = 0; kloc < Bsize; ++kloc)
         {
           accumulator += Awrk[iloc][kloc] * Bwrk[kloc][jloc];
         }
         // sycl::group_barrier(idx.get_group());
         idx.barrier();
       }
       c[i][j] = accumulator;
     });
   })
    .wait();
}

void init_input_matrices(sycl::queue &Q, const size_t Ndim, const size_t Mdim, const size_t Pdim, sycl::buffer<double, 2> &A, sycl::buffer<double, 2> &B)
{

  // Initilise A
  Q.submit([&](sycl::handler &cgh) {
    // sycl::accessor a {A, cgh, sycl::write_only, sycl::noinit};
    auto a = A.get_access<sycl::access_mode::write>(cgh);

    cgh.parallel_for(sycl::range<2>{Ndim, Pdim}, [=](sycl::id<2> idx) {
      a[idx] = Aval * static_cast<double>(idx[1] + 1);
    });
  });

  // Initilise B
  Q.submit([&](sycl::handler &cgh) {
    // sycl::accessor b {B, cgh, sycl::write_only, sycl::noinit};
    auto b = B.get_access<sycl::access_mode::write>(cgh);

    cgh.parallel_for(sycl::range<2>{Pdim, Mdim}, [=](sycl::id<2> idx) {
      b[idx] = static_cast<double>(idx[1] + 1) * Bval * static_cast<double>(idx[0] + 1);
    });
  });
}

void get_true_solution(const int Ndim, const int Mdim, const int Pdim, double *C)
{

  // Calculated from sum of k squared for k = 1 to P
  // Scale by AVAL and BVAL factors and column scaling of B
  double Ctmp = static_cast<double>(Pdim);
  double Cval = Ctmp * (Ctmp + 1.0) * (2.0 * Ctmp + 1.0);
  Cval = Cval * Aval * Bval / 6.0;

  for(int i = 0; i < Ndim; ++i)
  {
    for(int j = 0; j < Mdim; ++j)
    {
      C[i * Mdim + j] = Cval * static_cast<double>(j + 1);
    }
  }
}

// Return the sum of the squares of the differences of the two input matrices
double error(const int Ndim, const int Mdim, double *C, double *Cgold)
{
  double err = 0.0;

  for(int i = 0; i < Ndim; ++i)
  {
    for(int j = 0; j < Mdim; ++j)
    {
      double diff = C[i * Mdim + j] - Cgold[i * Mdim + j];
      err += diff * diff;
    }
  }

  return err;
}

int main()
{
  constexpr size_t local_size = 16;
  constexpr size_t Mdim = 256;
  constexpr size_t Ndim = 256;
  constexpr size_t Pdim = 256;

  cl::sycl::queue queue;

  std::vector<double> Cgold(Ndim * Mdim);
  get_true_solution(Ndim, Mdim, Pdim, Cgold.data());

  // Allocate memory
  sycl::buffer<double, 2> A({Ndim, Pdim});
  sycl::buffer<double, 2> B({Pdim, Mdim});
  sycl::buffer<double, 2> C({Ndim, Mdim});

  init_input_matrices(queue, Ndim, Mdim, Pdim, A, B);

  matmul_blocked(queue, Ndim, Mdim, Pdim, A, B, C);

  double err = error(Ndim, Mdim, C.get_access<sycl::access_mode::read>().get_pointer(), Cgold.data());

  // CHECK: Solution correct
  if (err < 1.0E-8) {
    std::cout << "  Solution correct" << std::endl;
  } else {
    std::cout
      << "  Solution *NOT* correct" << std::endl
      << "    Error = " << err << std::endl;
  }
}
