rm -rf build
mkdir build
cd build
cmake .. -G Xcode -DCMAKE_PREFIX_PATH=/Users/cjstein/Code/micro-tcn/realtime/libtorch ..
cmake --build .
