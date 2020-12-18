#include<iostream>
#include<string>
#include<cstring>
#include<cstdint>
#include<torch/torch.h>
#include <torch/script.h>

#include <chrono> 
using namespace std::chrono; 

struct TCNModel : torch::nn::Module {
    TCNModel(int nInputs, 
          int nOutputs, 
          int nLayers, 
          int nChannels, 
          int kernelWidth, 
          bool bias, 
          float* dilations) {

        for (int i = 0; i < nLayers; i++) {
            if (i == 0) {
                inChannels = nInputs;
                outChannels = nChannels;
            }
            else if (i + 1 == nLayers) {
                inChannels = nChannels;
                outChannels = nOutputs;
            }
            else {
                inChannels = nChannels;
                outChannels = nChannels;
            }
            conv.push_back(torch::nn::Conv1d(
                torch::nn::Conv1dOptions(inChannels,outChannels,kernelWidth)
                .stride(1)
                .dilation(dilations[i])
                .bias(bias)));
        }

        // now register each convolutional layer
        for (auto i = 0; i < conv.size(); i++) {
            register_module("conv"+std::to_string(i), conv[i]);
        }

        // and setup the activation functions
        leakyrelu = torch::nn::LeakyReLU(
                        torch::nn::LeakyReLUOptions()
                        .negative_slope(0.2));
    }
    // the forward operation
    torch::Tensor forward(torch::Tensor x, torch::Tensor p) {
        // we iterate over the convolutions
        for (auto i = 0; i < conv.size(); i++) {
          if (i + 1 < conv.size()) {
            x = leakyrelu(conv[i](x));
          }
          else {
              x = conv[i](x);
          }
        }
        return x;
    }

    void init(std::string initType){
        for (auto i = 0; i < conv.size(); i++) {
            if (initType.compare("normal"))
                torch::nn::init::normal_(conv[i]->weight);
            else if (initType.compare("uniform"))
                torch::nn::init::normal_(conv[i]->weight, -0.25, 0.25);
            else if (initType.compare("xavier_normal"))
                torch::nn::init::xavier_normal_(conv[i]->weight);
            else if (initType.compare("xavier_uniform"))
                torch::nn::init::xavier_uniform_(conv[i]->weight);
            else if (initType.compare("kaiming_normal"))
                torch::nn::init::kaiming_normal_(conv[i]->weight);
            else if (initType.compare("kamming_uniform"))
                torch::nn::init::kaiming_uniform_(conv[i]->weight);            
        }
    }

    int inChannels, outChannels;
    std::vector<torch::nn::Conv1d> conv;
    torch::nn::LeakyReLU leakyrelu;
};

int main(int argc, const char* argv[]){

    // don't compute gradients
    torch::NoGradGuard no_grad_guard; 

    // define the model config
    int nInputs     = 1;
    int nOutputs    = 1;
    int nLayers     = 10;
    int nChannels   = 32;
    int kernelWidth = 3;
    bool bias       = true;
    float dilations [10] = {1,2,4,8,16,32,64,128,256,512};

    // build the model
    TCNModel model(nInputs, 
                nOutputs, 
                nLayers, 
                nChannels, 
                kernelWidth, 
                bias, 
                dilations);

    if (argc != 2) {
        std::cerr << "usage: realtime <path-to-exported-script-module>\n";
        return -1;
    }

    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    std::cout << "ok\n";

    // pass through a random tensor
    auto out_duration = 44100;
    auto x_vec = torch::rand({1,1,44100});
    auto p_vec = torch::rand({1,1,2});

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(x_vec);
    inputs.push_back(p_vec);

    std::vector<double> runs;

    for (auto i = 0; i < 100; i++){
      auto start = high_resolution_clock::now(); 
      auto out = module.forward(inputs);
      auto stop = high_resolution_clock::now(); 
      auto duration = duration_cast<microseconds>(stop - start);
      runs.push_back(duration.count());
    }

    // compute the mean runtime
    double sum = std::accumulate(runs.begin(), runs.end(), 0.0);
    double mean_duration = sum / runs.size();

    auto elapsed_ms = mean_duration / 1000;
    auto rtf = (out_duration/44.1) / elapsed_ms;
    std::cout << elapsed_ms << " ms" << " RTF " << rtf << "x" << std::endl; 

    //for (auto i = 0; i < out.dim(); i++){
    //    std::cout << out.size(i) << std::endl;
    //}

}