rm -rf build
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/Users/cjstein/Code/micro-tcn/realtime/libtorch ..
cmake --build . --config Release
./realtime ../../models/traced_lstm_model.pt