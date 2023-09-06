# Compile and run
Remember to load CUDA on the system path "PATH" and "LD_LIBRARY_PATH".

```bash
mkdir build
cd build
cmake -G Ninja -DCMAKE_CXX_COMPILER=nvcc ..
cmake --build .
./tvl1 --camera=0
```

## Acknowledgement
This work has been supported by the EU (FEDER), the Spanish MINECO and CM under grants S2018/TCS-4423, PID2021-126576NB-I00 funded by MCIN/AEI/10.13039/501100011033 and by "ERDF A way of making Europe".