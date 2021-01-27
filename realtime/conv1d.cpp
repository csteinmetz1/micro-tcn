#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <chrono> 
using namespace std;
using namespace std::chrono; 

vector<vector<float>> conv1d(vector<vector<float>> x, vector<vector<vector<float>>> w) {
    /**
     * Compute the 1D convolution between w and x. 
     *
     * @param x Input sequence.             (in channels, samples).
     * @param w Convolutional filters.      (out channels, in channels, kernel).
     * @param b Convolutional bias terms.   (out channels)
     * 
     * 
     * @return y Output sequence.           (out channels, samples)
     */

    vector<vector<float>> y;

    auto in_ch = x.size();
    auto out_ch = w.size();
    auto k = w[0][0].size();

    cout << in_ch << " " << out_ch << endl;

    for (int c_o = 0; c_o < out_ch; c_o++){
        y.push_back(vector<float>());
        for (int c_i = 0; c_i < in_ch; c_i++){
            for (int x_i = 0; x_i < x[c_i].size() - (k - 1); x_i++) {
                if (c_i == 0) 
                    y[c_o].push_back(0.0);
                for (int w_i = 0; w_i < k; w_i++){
                    y[c_o][x_i] += x[c_i][x_i + w_i] * w[c_o][c_i][w_i];
                }
            }
        }
    }
    return y;
}

int main() {    

    int N = 128;
    int sr = 44100;
    int n_iters = 1000;

    /*
    vector<vector<float>> x
    {
        {0.2, 0.4, -0.3, 1.0, 0.1, 0.3, 0.2},
        {0.3, 0.2,  0.1, -0.1, 0.1, 0.3, 0.2},
    };

    vector<vector<vector<float>>> w 
    {
        {
            {0.0, 1.0, 0.0},
            {0.0, 0.3, 0.0},
        },
        {
            {0.0, 0.2, 0.1},
            {0.0, 0.1, 0.0},
        },
        {
            {1.0, 0.2, 0.1},
            {0.2, 0.1, 0.0},
        },
        {
            {0.9, 0.2, 0.1},
            {0.0, 0.1, 0.1},
        }
    };
    */

    vector<vector<float>> x(2, std::vector<float>(44100, 0));
    vector<vector<vector<float>>> w(32, vector<vector<float>>(2, vector<float>(13)));

    vector<vector<float>> y;

    auto start = high_resolution_clock::now(); 
    y = conv1d(x, w);
    auto stop = high_resolution_clock::now(); 

    auto duration = duration_cast<microseconds>(stop - start); 

    cout << duration.count() / (float)1e6 << "sec" << endl;
    
    /*
    for (int n = 0; n < 4; n++){
        cout << y[n].size() << " ";
        for (int i = 0; i < y[n].size(); i++){
            cout << y[n][i] << " ";
        }
        cout << endl;
    }
    */

    //
    //for (int i = 0; i < n_iters; i++) {
    //    conv1d(x, w);
    //}
    //
    //
    //auto rt = (); ///(float)n_iters);
    //cout << rt << "x real-time" << endl; 


}