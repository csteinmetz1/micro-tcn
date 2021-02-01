#ifndef UTCNLIB_H
#define UTCNLIB_H

#include <torch/torch.h>

struct Model : public torch::nn::Module {

    public:

        enum Activation {Linear, LeakyReLU, Tanh, Sigmoid, ReLU, ELU, SELU, GELU, RReLU, Softplus, Softshrink, Sine, Sine30};
        enum InitType   {normal, uniform1, uniform2, xavier_normal, xavier_uniform, kaiming_normal, kamming_uniform};

        Model(int nInputs, 
              int nOutputs, 
              int nLayers, 
              int nChannels, 
              int kWidth, 
              int dilationFactor,
              bool useBias, 
              int act,
              int init,
              int seed,
              bool dwise);

        torch::Tensor forward(torch::Tensor);
        void initModel(int seed);
        void buildModel(int seed);
        int getOutputSize(int frameSize);
        int getNumParameters();

        void setBias(bool newBias){bias = newBias;};
        void setInputs(int newInputs){inputs = newInputs;};
        void setLayers(int newLayers){layers = newLayers;};
        void setOutputs(int newOutputs){outputs = newOutputs;};
        void setChannels(int newChannels){channels = newChannels;};
        void setActivation(Activation newActivation){activation = newActivation;};
        void setInitType(InitType newInitType){initType = newInitType;};
        void setKernelWidth(int newKernelWidth){kernelWidth = newKernelWidth;};
        void setDilationFactor(int newDilationFactor){dilationFactor = newDilationFactor;};

        bool getBias(){return bias;};
        int getInputs(){return inputs;};
        int getLayers(){return layers;};
        int getOutputs(){return outputs;};
        int getChannels(){return channels;};
        int getKernelWidth(){return kernelWidth;};
        int getDilationFactor(){return dilationFactor;};
        Activation getActivation(){return activation;};
        InitType getInitType(){return initType;}

    private:
        int inputs, outputs, layers, channels, kernelWidth, dilationFactor;
        bool bias, depthwise;
        Activation activation;
        InitType initType;
        std::vector<torch::nn::Conv1d> conv;      
        torch::nn::LeakyReLU leakyrelu;
};

#endif