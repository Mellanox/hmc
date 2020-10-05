# Hardware MultiCast library

## Build
```sh
./autogen.sh
./configure --prefix=/where/to/install --with-ucx=<path_to_ucx>
make -j instal
```

## Build & run example
Requires mpicc/mpirun
```
cd example; make
mpirun -x HMC_VERBOSE=5 -x HMC_NET_DEVICES=mlx5_0:1 -np <NP> ./hmc_ibcast
```

#Build with cuda
```sh
./autogen.sh
./configure --prefix=/where/to/install --with-ucx=<path_to_ucx>  --with-cuda=<path_to_cuda>
make -j instal
```
