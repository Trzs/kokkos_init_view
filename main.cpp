#include <iostream>
#include <Kokkos_Core.hpp>

using std::cout;
using std::endl;
using Kokkos::View;
using Kokkos::fence;

constexpr int size = 3000 * 3000;

int main() {

    Kokkos::initialize(Kokkos::InitializationSettings().set_device_id(0));
    {
        Kokkos::Timer timer;

        timer.reset();
        auto view_host = View<double*, Kokkos::HostSpace>("view_host", size);
        fence();
        cout << "T[HostSpace] = " << timer.seconds() << "sec" << endl;

        timer.reset();
        auto view_default = View<double*, Kokkos::DefaultExecutionSpace::memory_space>("view_default", size);
        fence();
        cout << "T[DefaultExecutionSpace] = " << timer.seconds() << "sec" << endl;

        timer.reset();
        auto view_shared = View<double*, Kokkos::SharedSpace>("view_shared", size);
        fence();
        cout << "T[SharedSpace] = " << timer.seconds() << "sec" << endl;

        timer.reset();
        auto view_sharedhostpinned = View<double*, Kokkos::SharedHostPinnedSpace>("view_sharedhostpinned", size);
        fence();
        cout << "T[SharedHostPinnedSpace] = " << timer.seconds() << "sec" << endl;

#ifdef KOKKOS_ENABLE_HIP
        auto view_hipmanaged = View<double*, Kokkos::HIPManagedSpace>("view_hipmanaged", size);
        fence();
        cout << "T[HIPManaged] = " << timer.seconds() << "sec" << endl;

        timer.reset();
        auto view_hipspace = View<double*, Kokkos::HIPSpace>("view_hipspace", size);
        fence();
        cout << "T[HIP] = " << timer.seconds() << "sec" << endl;
#endif

#ifdef KOKKOS_ENABLE_CUDA
        auto view_cudauvm = View<double*, Kokkos::CudaUVMSpace>("view_cudauvm", size);
        fence();
        cout << "T[CudaUVM] = " << timer.seconds() << "sec" << endl;

        timer.reset();
        auto view_cudaspace = View<double*, Kokkos::CudaSpace>("view_cudaspace", size);
        fence();
        cout << "T[CUDA] = " << timer.seconds() << "sec" << endl;
#endif
    }
    Kokkos::finalize();

}
