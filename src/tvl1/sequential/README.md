# Compile and run
First of all, load the oneAPI variables:

```bash
source /path/to/oneapi/setvars.sh
```

```bash
mkdir build
cd build
cmake -G Ninja -DCMAKE_CXX_COMPILER=icpx ..
cmake --build .
./tvl1 --camera=0
```

## Acknowledgement
This work has been supported by the EU (FEDER), the Spanish MINECO and CM under grants S2018/TCS-4423, PID2021-126576NB-I00 funded by MCIN/AEI/10.13039/501100011033 and by "ERDF A way of making Europe".