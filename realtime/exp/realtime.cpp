#include<iostream>
#include<string>
#include<cstring>
#include<cstdint>
#include<torch/torch.h>
#include <torch/script.h>

#include <chrono> 
using namespace std::chrono; 

int main(int argc, const char* argv[]){

    // don't compute gradients
    torch::NoGradGuard no_grad_guard; 

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