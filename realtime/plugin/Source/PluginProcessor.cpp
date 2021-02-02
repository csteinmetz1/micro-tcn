/*
  ==============================================================================

    This file was auto-generated!

    It contains the basic framework code for a JUCE plugin processor.

  ==============================================================================
*/

#include "PluginProcessor.h"
#include "PluginEditor.h"
//#include "utcnlib.h"
#include <torch/script.h>
#include <torch/torch.h>

using namespace torch::indexing;

//==============================================================================
uTCNAudioProcessor::uTCNAudioProcessor()
#ifndef JucePlugin_PreferredChannelConfigurations
     : AudioProcessor (BusesProperties()
                     #if ! JucePlugin_IsMidiEffect
                      #if ! JucePlugin_IsSynth
                       .withInput  ("Input",  AudioChannelSet::stereo(), true)
                      #endif
                       .withOutput ("Output", AudioChannelSet::stereo(), true)
                     #endif
                       ),
#endif
    parameters (*this, nullptr, Identifier ("ronn"),
    {
        std::make_unique<AudioParameterFloat> ("inputGain", "Input Gain", -24.0f, 24.0f, 0.0f),   
        std::make_unique<AudioParameterFloat> ("outputGain", "Output Gain", -24.0f, 24.0f, 0.0f),

        std::make_unique<AudioParameterFloat> ("limit", "Limit/Compress", 0.0f, 1.0f, 0.0f),
        std::make_unique<AudioParameterFloat> ("peakReduction", "Peak Reduction", 0.0f, 100.0f, 50.0f),
    })
{
 
    inputGainParameter     = parameters.getRawParameterValue ("inputGain");
    outputGainParameter    = parameters.getRawParameterValue ("outputGain");
    limitParameter         = parameters.getRawParameterValue ("limit");
    peakReductionParameter = parameters.getRawParameterValue ("peakReduction");

    // neural network model
    buildModel();
}

uTCNAudioProcessor::~uTCNAudioProcessor()
{
    // we may need to delete the model here
}

//==============================================================================
const String uTCNAudioProcessor::getName() const
{
    return JucePlugin_Name;
}

bool uTCNAudioProcessor::acceptsMidi() const
{
   #if JucePlugin_WantsMidiInput
    return true;
   #else
    return false;
   #endif
}

bool uTCNAudioProcessor::producesMidi() const
{
   #if JucePlugin_ProducesMidiOutput
    return true;
   #else
    return false;
   #endif
}

bool uTCNAudioProcessor::isMidiEffect() const
{
   #if JucePlugin_IsMidiEffect
    return true;
   #else
    return false;
   #endif
}

double uTCNAudioProcessor::getTailLengthSeconds() const
{
    return 0.0;
}

int uTCNAudioProcessor::getNumPrograms()
{
    return 1;   // NB: some hosts don't cope very well if you tell them there are 0 programs,
                // so this should be at least 1, even if you're not really implementing programs.
}

int uTCNAudioProcessor::getCurrentProgram()
{
    return 0;
}

void uTCNAudioProcessor::setCurrentProgram (int index)
{
}

const String uTCNAudioProcessor::getProgramName (int index)
{
    return {};
}

void uTCNAudioProcessor::changeProgramName (int index, const String& newName)
{
}

//==============================================================================
void uTCNAudioProcessor::prepareToPlay (double sampleRate_, int samplesPerBlock_)
{
    // store the sample rate for future calculations
    sampleRate = sampleRate_;
    blockSamples = samplesPerBlock_;

    // setup high pass filter model
    double freq = 10.0;
    double q = 10.0;
    for (int channel = 0; channel < getTotalNumOutputChannels(); ++channel) {
        IIRFilter filter;
        filter.setCoefficients(IIRCoefficients::makeHighPass (sampleRate_, freq, q));
        highPassFilters.push_back(filter);
    }

    calculateReceptiveField();      // compute the receptive field, make sure it's up to date
    setupBuffers();                 // setup the buffer for handling context
}

void uTCNAudioProcessor::releaseResources()
{
    // When playback stops, you can use this as an opportunity to free up any
    // spare memory, etc.
}

#ifndef JucePlugin_PreferredChannelConfigurations
bool uTCNAudioProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
  #if JucePlugin_IsMidiEffect
    ignoreUnused (layouts);
    return true;
  #else
    // This is the place where you check if the layout is supported.
    // In this template code we only support mono.
    if (layouts.getMainOutputChannelSet() != AudioChannelSet::mono())
        return false;

    // This checks if the input layout matches the output layout
   #if ! JucePlugin_IsSynth
    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;
   #endif

    return true;
  #endif
}
#endif

void uTCNAudioProcessor::calculateReceptiveField()
{
    /*
    int k = *kernelParameter;
    int d = *dilationParameter;
    int l = *layersParameter;
    double rf =  k * d;

    for (int layer = 1; layer < l; ++layer) {
        rf = rf + ((k-1) * pow(d,layer));
    }
    */

    receptiveFieldSamples = 13333; // store in attribute
}

void uTCNAudioProcessor::setupBuffers()
{
    // compute the size of the buffer which will be passed to model
    membuflength = (int)(receptiveFieldSamples - 1);
    procbuflength = (int)(receptiveFieldSamples - 1 + blockSamples);

    std::cout << "membuflength " << membuflength << std::endl;
    std::cout << "procbuflength " << procbuflength << std::endl;

    // Initialize the to n channels
    nInputs = getTotalNumInputChannels();

    // and membuflength samples per channel
    membuf.setSize(1, membuflength);
    membuf.clear();

    procbuf.setSize(1, procbuflength);
    procbuf.clear();
}

void uTCNAudioProcessor::processBlock (AudioBuffer<float>& buffer, MidiBuffer& midiMessages)
{
    ScopedNoDenormals noDenormals;
    auto inChannels  = getTotalNumInputChannels();
    auto outChannels = getTotalNumOutputChannels();
    
    // we have to handle some buffer business first (this is somewhat inefficient)
    // 1. first we construct the process buffer which is [membuf, buffer]
    procbuf.copyFrom(0,0,membuf,0,0,membuflength);              // first copy the past samples into the process buffer
    procbuf.copyFrom(0,membuflength,buffer,0,0,blockSamples);   // second copy the current buffer samples at the end

    // 2. now we update membuf to reflect the last N samples in proccess buffer
    membuf.copyFrom(0,0,procbuf,0,procbuflength-membuflength,membuflength); 

    // 3. now move the process buffer to a tensor
    std::vector<int64_t> sizes = {procbuflength*inChannels};// size of the process buffer data
    auto* procbufptr = procbuf.getWritePointer(0);          // get pointer of the first channel 
    at::Tensor frame = torch::from_blob(procbufptr, sizes); // load data from buffer into tensor type

    frame = torch::mul(frame, inputGainLn);                 // apply the input gain first
    frame = torch::reshape(frame, {1,1,procbuflength});     // reshape so we have a batch and channel dimension

    at::Tensor parameters = torch::empty({2});
    parameters.index_put_({0}, (float)*limitParameter);
    parameters.index_put_({1}, (float)*peakReductionParameter/100.0);
    parameters = torch::reshape(parameters, {outChannels,1,2});      // reshape so we have a batch and channel dimension

    std::vector<torch::jit::IValue> inputs;                 // create special holder for model inputs
    inputs.push_back(frame);                                // add the process buffer
    inputs.push_back(parameters);                           // add the parameter values (conditioning)

    at::Tensor output = model.forward(inputs).toTensor();

    // now load the output channels back into the buffer
    for (int channel = 0; channel < outChannels; ++channel) {
        auto outputData = output.index({channel,0,torch::indexing::Slice()});      // index the proper output channel
        auto outputDataPtr = outputData.data_ptr<float>();      
        buffer.copyFrom(channel,0,outputDataPtr,blockSamples);    // copy output data to buffer
        // remove the DC bias                                      
        highPassFilters[channel].processSamples(buffer.getWritePointer (channel), buffer.getNumSamples());
    }
    buffer.applyGain(outputGainLn);                                  // apply the output gain
   
}

//==============================================================================
bool uTCNAudioProcessor::hasEditor() const
{
    return true; // (change this to false if you choose to not supply an editor)
}

AudioProcessorEditor* uTCNAudioProcessor::createEditor()
{
    return new uTCNAudioProcessorEditor (*this, parameters);
}

//==============================================================================
void uTCNAudioProcessor::getStateInformation (MemoryBlock& destData)
{
    auto state = parameters.copyState();
    std::unique_ptr<XmlElement> xml (state.createXml());
    copyXmlToBinary (*xml, destData);
}

void uTCNAudioProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    std::unique_ptr<XmlElement> xmlState (getXmlFromBinary (data, sizeInBytes));
 
    if (xmlState.get() != nullptr)
        if (xmlState->hasTagName (parameters.state.getType()))
            parameters.replaceState (ValueTree::fromXml (*xmlState));
}

//==============================================================================

void uTCNAudioProcessor::buildModel() 
{
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        model = torch::jit::load("/Users/cjstein/Code/micro-tcn/models/traced_1-uTCN-300__causal__4-10-13__fraction-0.01-bs32.pt");
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    std::cout << "ok\n";
    calculateReceptiveField();
    std::cout << "receptive field: " << receptiveFieldSamples << std::endl;;
    setupBuffers();

}

//==============================================================================
// This creates new instances of the plugin..
AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new uTCNAudioProcessor();
}
