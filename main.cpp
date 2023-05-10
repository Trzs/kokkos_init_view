#include <iostream>
#include <string>
#include <Kokkos_Core.hpp>

using std::cout;
using std::endl;
using Kokkos::View;
using Kokkos::fence;

constexpr int size = 3000*3000;
constexpr int repeats = 2;

template <class SPACE>
void init_view(std::string name, int repeats, int length) {
        for (int i=0; i<repeats; ++i) {
                auto view_host = View<double*, SPACE>(name, length);
                fence();
        }
}

int main() {

    Kokkos::initialize(Kokkos::InitializationSettings().set_device_id(0));
    {
        Kokkos::Timer timer;

        cout << "Init size: " << size << " elements." << endl;
        cout << "Average time from " << repeats << " repeats." << endl;

        timer.reset();
        init_view<Kokkos::HostSpace>("view_host", repeats, size);
        cout << "T[HostSpace] = " << timer.seconds()/repeats << "sec" << endl;

        timer.reset();
        init_view<Kokkos::DefaultExecutionSpace::memory_space>("view_default", repeats, size);
        cout << "T[DefaultExecutionSpace] = " << timer.seconds()/repeats << "sec" << endl;

        timer.reset();
        init_view<Kokkos::SharedSpace>("view_shared", repeats, size);
        cout << "T[SharedSpace] = " << timer.seconds()/repeats << "sec" << endl;

        timer.reset();
        init_view<Kokkos::SharedHostPinnedSpace>("view_sharedhostpinned", repeats, size);
        cout << "T[SharedHostPinnedSpace] = " << timer.seconds()/repeats << "sec" << endl;

#ifdef KOKKOS_ENABLE_HIP
        timer.reset();
        init_view<Kokkos::HIPManagedSpace>("view_hipmanaged", repeats, size);
        cout << "T[HIPManaged] = " << timer.seconds()/repeats << "sec" << endl;

        timer.reset();
        init_view<Kokkos::HIPSpace>("view_hipspace", repeats, size);
        cout << "T[HIP] = " << timer.seconds()/repeats << "sec" << endl;
#endif

#ifdef KOKKOS_ENABLE_CUDA
        timer.reset();
        init_view<Kokkos::CudaUVMSpace>("view_cudauvm", repeats, size);
        cout << "T[CudaUVM] = " << timer.seconds()/repeats << "sec" << endl;

        timer.reset();
        init_view<Kokkos::CudaSpace>("view_cudaspace", repeats, size);
        cout << "T[CUDA] = " << timer.seconds()/repeats << "sec" << endl;
#endif
    }
    Kokkos::finalize();
}
