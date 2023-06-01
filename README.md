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
Init size: 9000000 elements.
Average time from 2 repeats.
T[HostSpace] = 0.00597439sec
T[DefaultExecutionSpace] = 0.000385006sec
T[SharedSpace] = 0.00613634sec
T[SharedHostPinnedSpace] = 0.0214827sec
T[CudaUVM] = 0.00532685sec
T[CUDA] = 0.000289473sec
```
