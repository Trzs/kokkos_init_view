# kokkos_init_view

## build
```
mkdir build
cd build
cmake ..
make -j 4
```

## run
```
user@system:build$ ./kokkos_init_view
T[HostSpace] = 0.0150688sec
T[DefaultExecutionSpace] = 0.000488684sec
T[SharedSpace] = 0.0069286sec
T[SharedHostPinnedSpace] = 0.0208152sec
T[CudaUVM] = 0.0277169sec
T[CUDA] = 0.000306112sec
```
