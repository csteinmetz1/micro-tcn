/*
  ==============================================================================

    This file was auto-generated!

    It contains the basic framework code for a JUCE plugin editor.

  ==============================================================================
*/

#pragma once

#include <JuceHeader.h>
#include "PluginProcessor.h"

//==============================================================================
/**
*/
class uTCNAudioProcessorEditor  : public AudioProcessorEditor
{
public:
    enum
    {
        paramControlHeight = 40,
        paramLabelWidth    = 80,
        paramSliderWidth   = 300
    };

    uTCNAudioProcessorEditor (uTCNAudioProcessor&, AudioProcessorValueTreeState&);
    ~uTCNAudioProcessorEditor();

    typedef AudioProcessorValueTreeState::SliderAttachment SliderAttachment;
    typedef AudioProcessorValueTreeState::ButtonAttachment ButtonAttachment;
    typedef AudioProcessorValueTreeState::ComboBoxAttachment ComboBoxAttachment;


    //==============================================================================
    void paint (Graphics&) override;
    void resized() override;
    void updateModelState();
    void updateGains(bool inputGain);

private:
    // This reference is provided as a quick way for your editor to
    // access the processor object that created it.
    uTCNAudioProcessor& processor;
    AudioProcessorValueTreeState& valueTreeState;

    // Main panel controls
    //==============================================================================

    Slider layersSlider, kernelSlider, channelsSlider;
    Label layersLabel, kernelLabel, channelsLabel;
    std::unique_ptr<SliderAttachment> layersAttachment, kernelAttachment, channelsAttachment;
 
    ToggleButton useBiasButton, linkGainButton, depthwiseButton; 
    std::unique_ptr<ButtonAttachment> useBiasAttachment, linkGainAttachment, depthwiseAttachment;

    ComboBox dilationsComboBox, activationsComboBox, initTypeComboBox;
    Label dilationsLabel, activationsLabel, initTypeLabel;
    std::unique_ptr<ComboBoxAttachment> dilationsAttachment, activationsAttachment, initTypeAttachment;

    // Side panel controls
    //==============================================================================
    Label inputGainLabel, outputGainLabel;
    Slider inputGainSlider, outputGainSlider;
    std::unique_ptr<SliderAttachment> inputGainAttachment, outputGainAttachment;

    TextEditor receptiveFieldTextEditor, seedTextEditor, parametersTextEditor;
    Label receptiveFieldLabel, seedLabel, parametersLabel;
    String receptiveFieldString, seedString;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (uTCNAudioProcessorEditor)
};
