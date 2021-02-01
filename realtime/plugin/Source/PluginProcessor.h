/*
  ==============================================================================

    This file was auto-generated!

    It contains the basic framework code for a JUCE plugin processor.

  ==============================================================================
*/

#pragma once

#include <JuceHeader.h>
#include <torch/script.h>
#include <torch/torch.h>

//#include "utcnlib.h"

//==============================================================================
/**
*/
class uTCNAudioProcessor  : public AudioProcessor
{
public:
    //==============================================================================
    uTCNAudioProcessor();
    ~uTCNAudioProcessor();

    //==============================================================================
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

   #ifndef JucePlugin_PreferredChannelConfigurations
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
   #endif

    void processBlock (AudioBuffer<float>&, MidiBuffer&) override;

    //==============================================================================
    AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    //==============================================================================
    const String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    //==============================================================================
    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram (int index) override;
    const String getProgramName (int index) override;
    void changeProgramName (int index, const String& newName) override;

    //==============================================================================
    void getStateInformation (MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    //==============================================================================
    void calculateReceptiveField();
    void setupBuffers();

    //==============================================================================
    AudioParameterInt* layers;

    //==============================================================================
    void buildModel();

    int seed = 42;
    int receptiveFieldSamples = 0; // in samples
    int blockSamples = 0; // in/out samples
    double sampleRate = 0; // in Hz

    // holder for the linear gain values
    // (don't want to convert dB -> linear on audio thread)
    float inputGainLn, outputGainLn;

private:
    //==============================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (uTCNAudioProcessor)

    //==============================================================================
    AudioProcessorValueTreeState parameters;

    // is this not dangerous to use floats for values that are actually ints?
    std::atomic<float>* inputGainParameter  = nullptr;
    std::atomic<float>* outputGainParameter = nullptr;
    std::atomic<float>* layersParameter     = nullptr;
    std::atomic<float>* channelsParameter   = nullptr;
    std::atomic<float>* kernelParameter     = nullptr;
    std::atomic<float>* useBiasParameter    = nullptr;
    std::atomic<float>* dilationParameter   = nullptr;
    std::atomic<float>* activationParameter = nullptr;
    std::atomic<float>* initTypeParameter   = nullptr;
    std::atomic<float>* seedParameter       = nullptr;
    std::atomic<float>* depthwiseParameter  = nullptr;


    AudioBuffer<float> membuf, procbuf; // circular buffer to store input data
    int mbr, mbw; // read and write pointers

    int nInputs;
    int membuflength; // number of samples in the memory (past samples) buffer (rf - 1)
    int procbuflength; // number of samples in the process buffer (rf + block - 1)

    std::vector<IIRFilter> highPassFilters; // high pass filters for the left and right channels

    torch::jit::script::Module model;
};
