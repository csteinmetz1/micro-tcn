rm -rf build
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/home/cjstein/code/micro-tcn/realtime/libtorch ..
cmake --build . --config Release
./realtime ../../models/traced_1-uTCN-300__causal__4-10-13__fraction-0.01-bs32.pt