#include <iostream>
#include <string>
#include <cstring>
#include <cstdint>
#include <torch/torch.h>

#include "utcnlib.h"

Model::Model(int nInputs, 
             int nOutputs, 
             int nLayers, 
             int nChannels, 
             int kWidth,
             int dFactor, 
             bool useBias, 
             int act,
             int init,
             int seed,
             bool dwise) {

        inputs = nInputs;
        outputs = nOutputs;
        layers = nLayers;
        channels = nChannels;
        kernelWidth = kWidth;
        bias = useBias;
        activation = static_cast<Activation>(int(act));
        initType = static_cast<InitType>(int(init));
        dilationFactor = dFactor;
        depthwise = dwise;

        buildModel(seed);

        // and setup the activation functions
        leakyrelu = torch::nn::LeakyReLU(
                        torch::nn::LeakyReLUOptions()
                        .negative_slope(0.2));
}

void Model::buildModel(int seed) {

    
}

// the forward operation
torch::Tensor Model::forward(torch::Tensor x) {
    // we iterate over the convolutions
    for (auto i = 0; i < getLayers(); i++) {
        if (i + 1 < getLayers()) {
            //setActivation(static_cast<Activation>(rand() % Sine));
            switch (getActivation()) {
                case Linear:        x =                   (conv[i](x)); break;
                case LeakyReLU:     x = leakyrelu         (conv[i](x)); break;
                case Tanh:          x = torch::tanh       (conv[i](x)); break;
                case Sigmoid:       x = torch::sigmoid    (conv[i](x)); break;
                case ReLU:          x = torch::relu       (conv[i](x)); break;
                case ELU:           x = torch::elu        (conv[i](x)); break;
                case SELU:          x = torch::selu       (conv[i](x)); break;
                case GELU:          x = torch::gelu       (conv[i](x)); break;
                case RReLU:         x = torch::rrelu      (conv[i](x)); break;
                case Softplus:      x = torch::softplus   (conv[i](x)); break;
                case Softshrink:    x = torch::softshrink (conv[i](x)); break;
                case Sine:          x = torch::sin        (conv[i](x)); break;
                case Sine30:        x = torch::sin        (30 * conv[i](x)); break;
                default:            x =                   (conv[i](x)); break;
            }
        }
        else
            x = conv[i](x);
    }
    return x;
}

void Model::initModel(int seed){
    torch::manual_seed(seed); // always reset the seed before init
    for (auto i = 0; i < getLayers(); i++) {
        switch(getInitType())
        {
            case normal:            torch::nn::init::normal_            (conv[i]->weight);
            case uniform1:          torch::nn::init::uniform_           (conv[i]->weight, -0.25, 0.25);
            case uniform2:          torch::nn::init::uniform_           (conv[i]->weight, -1.00, 1.00);
            case xavier_normal:     torch::nn::init::xavier_normal_     (conv[i]->weight);
            case xavier_uniform:    torch::nn::init::xavier_uniform_    (conv[i]->weight);
            case kaiming_normal:    torch::nn::init::kaiming_normal_    (conv[i]->weight);
            case kamming_uniform:   torch::nn::init::kaiming_uniform_   (conv[i]->weight);
        }
    }
}

int Model::getOutputSize(int frameSize){
    int outputSize = frameSize;
    for (auto i = 0; i < getLayers(); i++) {
        outputSize = outputSize - ((getKernelWidth()-1) * pow(getDilationFactor(), i));
    }
    return outputSize;
}

int Model::getNumParameters(){
    int n = 0;
    for (const auto& p : parameters()) {
        auto sizes = p.sizes();
        int s = 1;
        for (auto dim : sizes) {
            std::cout << dim << std::endl;
            s = s * dim;
        }
        n = n + s;
    }
    return n;
}