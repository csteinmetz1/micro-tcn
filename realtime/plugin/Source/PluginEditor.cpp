/*
  ==============================================================================

    This file was auto-generated!

    It contains the basic framework code for a JUCE plugin editor.

  ==============================================================================
*/

#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
uTCNAudioProcessorEditor::uTCNAudioProcessorEditor (uTCNAudioProcessor& p, AudioProcessorValueTreeState& vts)
    : AudioProcessorEditor (&p), processor (p), valueTreeState (vts)
{

    getLookAndFeel().setColour (Slider::thumbColourId, Colours::grey);
    getLookAndFeel().setColour (Slider::trackColourId, Colours::lightgrey);
    getLookAndFeel().setColour (Slider::backgroundColourId, Colours::lightgrey);    
    getLookAndFeel().setColour (Slider::textBoxBackgroundColourId, Colours::white);
    getLookAndFeel().setColour (Slider::textBoxTextColourId, Colours::darkgrey);
    getLookAndFeel().setColour (Slider::textBoxHighlightColourId, Colours::darkgrey);
    getLookAndFeel().setColour (Slider::textBoxOutlineColourId, Colours::white);
    getLookAndFeel().setColour (Slider::rotarySliderFillColourId, Colours::lightgrey);
    getLookAndFeel().setColour (Slider::rotarySliderOutlineColourId, Colours::lightgrey);

    getLookAndFeel().setColour (ComboBox::backgroundColourId, Colours::white);
    getLookAndFeel().setColour (ComboBox::textColourId, Colours::darkgrey);
    getLookAndFeel().setColour (ComboBox::outlineColourId, Colours::lightgrey);
    getLookAndFeel().setColour (ComboBox::arrowColourId, Colours::darkgrey);

    getLookAndFeel().setColour (ToggleButton::textColourId, Colours::darkgrey);
    getLookAndFeel().setColour (ToggleButton::tickColourId, Colours::darkgrey);
    getLookAndFeel().setColour (ToggleButton::tickDisabledColourId, Colours::lightgrey);

    getLookAndFeel().setColour (Label::textColourId, Colours::darkgrey);

    getLookAndFeel().setColour (PopupMenu::backgroundColourId, Colours::white);
    getLookAndFeel().setColour (PopupMenu::textColourId, Colours::darkgrey);
    getLookAndFeel().setColour (PopupMenu::highlightedBackgroundColourId, Colours::lightgrey);
    getLookAndFeel().setColour (PopupMenu::highlightedTextColourId, Colours::darkgrey);

    useBiasButton.setButtonText ("Bias");
    linkGainButton.setButtonText ("Link");
    depthwiseButton.setButtonText ("Depthwise");

    Colour fillColour = Colour (0xffececec); // side panel color

    inputGainSlider.setRange(-24, 24);
    inputGainSlider.setSliderStyle (Slider::Rotary);
    inputGainSlider.setTextBoxStyle (Slider::TextBoxRight, false, 50, 24);//(Slider::NoTextBox, false, 0, 0);
    inputGainSlider.onValueChange = [this] {updateGains(true);};
    inputGainSlider.setValue (juce::Decibels::gainToDecibels(processor.inputGainLn));
    inputGainSlider.setColour (Slider::textBoxBackgroundColourId, fillColour);
    inputGainSlider.setColour (Slider::textBoxOutlineColourId, fillColour);
    inputGainLabel.setText ("in", dontSendNotification);
    inputGainLabel.attachToComponent (&inputGainSlider, true); 
    
    outputGainSlider.setRange(-24, 24);
    outputGainSlider.setSliderStyle (Slider::Rotary);
    outputGainSlider.setTextBoxStyle (Slider::TextBoxRight, false, 50, 24);//(Slider::NoTextBox, false, 0, 0);
    outputGainSlider.onValueChange = [this] {updateGains(false);};
    outputGainSlider.setValue (juce::Decibels::gainToDecibels(processor.outputGainLn));
    outputGainSlider.setColour (Slider::textBoxBackgroundColourId, fillColour);
    outputGainSlider.setColour (Slider::textBoxOutlineColourId, fillColour);
    outputGainLabel.setText ("out", dontSendNotification);
    outputGainLabel.attachToComponent (&outputGainSlider, true); 

    layersSlider.setTextBoxStyle (Slider::TextBoxRight, false, 30, 24);
    kernelSlider.setTextBoxStyle (Slider::TextBoxRight, false, 30, 24);
    channelsSlider.setTextBoxStyle (Slider::TextBoxRight, false, 30, 24);

    addAndMakeVisible (layersSlider);
    addAndMakeVisible (kernelSlider);
    addAndMakeVisible (channelsSlider);
    addAndMakeVisible (inputGainSlider);
    addAndMakeVisible (outputGainSlider);
    addAndMakeVisible (dilationsComboBox);
    addAndMakeVisible (activationsComboBox);
    addAndMakeVisible (initTypeComboBox);
    addAndMakeVisible (useBiasButton);
    addAndMakeVisible (linkGainButton);
    addAndMakeVisible (depthwiseButton);
    addAndMakeVisible (inputGainLabel);
    addAndMakeVisible (outputGainLabel);

    // attach labels
    addAndMakeVisible (layersLabel);
    layersLabel.setText ("layers", dontSendNotification);
    layersLabel.attachToComponent (&layersSlider, true); 
    addAndMakeVisible (kernelLabel);
    kernelLabel.setText ("kernel", dontSendNotification);
    kernelLabel.attachToComponent (&kernelSlider, true); 
    addAndMakeVisible (channelsLabel);
    channelsLabel.setText ("channels", dontSendNotification);
    channelsLabel.attachToComponent (&channelsSlider, true); 

    // add options to comboboxes
    dilationsComboBox.addItem ("1^n", 1);
    dilationsComboBox.addItem ("2^n", 2);
    dilationsComboBox.addItem ("3^n", 3);
    dilationsComboBox.addItem ("4^n", 4);

    activationsComboBox.addItem("Linear",     1);
    activationsComboBox.addItem("LeakyReLU",  2);
    activationsComboBox.addItem("Tanh",       3);
    activationsComboBox.addItem("Sigmoid",    4);
    activationsComboBox.addItem("ReLU",       5);
    activationsComboBox.addItem("ELU",        6);
    activationsComboBox.addItem("SELU",       7);
    activationsComboBox.addItem("GELU",       8);
    activationsComboBox.addItem("RReLU",      9);
    activationsComboBox.addItem("Softplus",  10);
    activationsComboBox.addItem("Softshrink",11); 
    activationsComboBox.addItem("Sine",      12); 
    activationsComboBox.addItem("Sine x 30", 13); 

    initTypeComboBox.addItem("Normal", 1);
    initTypeComboBox.addItem("Uniform (-0.25, 0.25)", 2);
    initTypeComboBox.addItem("Uniform (-1.00, 1.00)", 3);
    initTypeComboBox.addItem("Xavier (Normal)", 4);
    initTypeComboBox.addItem("Xavier (Uniform)", 5);
    initTypeComboBox.addItem("Kaiming (Normal)", 6);
    initTypeComboBox.addItem("Kaiming (Uniform)", 7);

    addAndMakeVisible (dilationsLabel);
    dilationsLabel.setText ("dilation", dontSendNotification);
    dilationsLabel.attachToComponent (&dilationsComboBox, true); 
    addAndMakeVisible (activationsLabel);
    activationsLabel.setText ("activation", dontSendNotification);
    activationsLabel.attachToComponent (&activationsComboBox, true); 
    addAndMakeVisible (initTypeLabel);
    initTypeLabel.setText ("init type", dontSendNotification);
    initTypeLabel.attachToComponent (&initTypeComboBox, true); 

    receptiveFieldTextEditor.setColour (TextEditor::backgroundColourId, fillColour);
    receptiveFieldTextEditor.setColour (TextEditor::outlineColourId, fillColour);
    receptiveFieldTextEditor.setColour (TextEditor::textColourId, Colours::darkgrey);
    receptiveFieldTextEditor.setColour (TextEditor::highlightColourId, Colours::darkgrey);
    receptiveFieldTextEditor.setReadOnly(true);
    receptiveFieldTextEditor.setFont(Font (15.0f));
    receptiveFieldTextEditor.setText("0", false);
    receptiveFieldLabel.setText ("receptive field", dontSendNotification);
    receptiveFieldLabel.attachToComponent (&receptiveFieldTextEditor, true); 
    addAndMakeVisible(receptiveFieldTextEditor);
    addAndMakeVisible(receptiveFieldLabel);

    parametersTextEditor.setColour (TextEditor::backgroundColourId, fillColour);
    parametersTextEditor.setColour (TextEditor::outlineColourId, fillColour);
    parametersTextEditor.setColour (TextEditor::textColourId, Colours::darkgrey);
    parametersTextEditor.setColour (TextEditor::highlightColourId, Colours::darkgrey);
    parametersTextEditor.setReadOnly(true);
    parametersTextEditor.setFont(Font (15.0f));
    parametersTextEditor.setText("0", false);
    parametersLabel.setText ("parameters", dontSendNotification);
    parametersLabel.attachToComponent (&parametersTextEditor, true); 
    addAndMakeVisible(parametersTextEditor);
    addAndMakeVisible(parametersLabel);

    seedTextEditor.setColour (TextEditor::backgroundColourId, fillColour);
    seedTextEditor.setColour (TextEditor::outlineColourId, fillColour);
    seedTextEditor.setColour (TextEditor::textColourId, Colours::darkgrey);
    seedTextEditor.setColour (TextEditor::highlightColourId, Colours::darkgrey);
    seedTextEditor.setReadOnly(false);
    seedTextEditor.setFont(Font (15.0f));
    seedTextEditor.setText("0", false);
    seedLabel.setText ("seed", dontSendNotification);
    seedLabel.attachToComponent (&seedTextEditor, true); 
    addAndMakeVisible(seedTextEditor);
    addAndMakeVisible(seedLabel);

    layersAttachment.reset      (new SliderAttachment   (valueTreeState, "layers", layersSlider));
    kernelAttachment.reset      (new SliderAttachment   (valueTreeState, "kernel", kernelSlider));
    channelsAttachment.reset    (new SliderAttachment   (valueTreeState, "channels", channelsSlider));
    inputGainAttachment.reset   (new SliderAttachment   (valueTreeState, "inputGain", inputGainSlider));
    outputGainAttachment.reset  (new SliderAttachment   (valueTreeState, "outputGain", outputGainSlider));
    dilationsAttachment.reset   (new ComboBoxAttachment (valueTreeState, "dilation", dilationsComboBox));
    activationsAttachment.reset (new ComboBoxAttachment (valueTreeState, "activation", activationsComboBox));
    initTypeAttachment.reset    (new ComboBoxAttachment (valueTreeState, "initType", initTypeComboBox));
    useBiasAttachment.reset     (new ButtonAttachment   (valueTreeState, "useBias", useBiasButton));
    linkGainAttachment.reset    (new ButtonAttachment   (valueTreeState, "linkGain", linkGainButton));
    depthwiseAttachment.reset   (new ButtonAttachment   (valueTreeState, "depthwise", depthwiseButton));
    //seedAttachment.reset        (new TextBoxAttachment  (valueTreeState, "seed", seedTextEditor));

    // callbacks for updating the model (not all parameters)
    layersSlider.onValueChange   = [this] { updateModelState(); };
    kernelSlider.onValueChange   = [this] { updateModelState(); };
    channelsSlider.onValueChange = [this] { updateModelState(); };
    dilationsComboBox.onChange   = [this] { updateModelState(); };
    activationsComboBox.onChange = [this] { updateModelState(); };
    initTypeComboBox.onChange    = [this] { updateModelState(); };
    useBiasButton.onStateChange  = [this] { updateModelState(); };
    depthwiseButton.onStateChange = [this] { updateModelState(); };

    setSize (600, 300);
}

uTCNAudioProcessorEditor::~uTCNAudioProcessorEditor()
{
}

//==============================================================================
void uTCNAudioProcessorEditor::updateGains(bool inputGain)
{
  if (inputGain == true){
    processor.inputGainLn = juce::Decibels::decibelsToGain((float) inputGainSlider.getValue());
    inputGainSlider.setValue (juce::Decibels::gainToDecibels(processor.inputGainLn));
    if (linkGainButton.getToggleState()) {
      float outputGaindB = -1 * inputGainSlider.getValue();
      processor.outputGainLn = juce::Decibels::decibelsToGain((float) outputGaindB);
      outputGainSlider.setValue (juce::Decibels::gainToDecibels(processor.outputGainLn));
    }
  }
  else {
    processor.outputGainLn = juce::Decibels::decibelsToGain((float) outputGainSlider.getValue());
    outputGainSlider.setValue (juce::Decibels::gainToDecibels(processor.outputGainLn));
    if (linkGainButton.getToggleState()) {
      float inputGaindB = -1 * outputGainSlider.getValue();
      processor.inputGainLn = juce::Decibels::decibelsToGain((float) inputGaindB);
      inputGainSlider.setValue (juce::Decibels::gainToDecibels(processor.inputGainLn));
    }
  }
}

//==============================================================================
void uTCNAudioProcessorEditor::updateModelState()
{
  processor.calculateReceptiveField();
  float rfms = (processor.receptiveFieldSamples / processor.sampleRate) * 1000;
  receptiveFieldTextEditor.setText(String(rfms, 1));
  //int parameters = processor.model->getNumParameters();
  //parametersTextEditor.setText(String(parameters));
}

//==============================================================================
void uTCNAudioProcessorEditor::paint (Graphics& g)
{
    // fill the whole window white
    g.fillAll (Colours::white);

    // set the font size and draw text to the screen
    g.setFont (15.0f);

    // fill the right panel with grey
    {
      Colour fillColour = Colour (0xffececec);
      g.setColour (fillColour);
      g.fillRect (400, 0, 300, 300); // draw side bar on the right
      g.fillRect (0, 0, 30, 300);    // draw strip on the left

      g.setColour (Colours::grey);
      g.setFont (Font ("Source Sans Variable", 32.0f, Font::plain).withTypefaceStyle ("Light")); //.withExtraKerningFactor (0.147f));
      g.drawText ("ronn", 350, 0, 300, 70, Justification::centred, true);
      g.setFont (Font ("Source Sans Variable", 10.0f, Font::plain).withTypefaceStyle ("Light")); //.withExtraKerningFactor (0.147f));
      g.drawText ("randomized overdrive", 350, 0, 300, 105, Justification::centred, true);
      g.drawText ("neural networks", 350, 0, 300, 125, Justification::centred, true);
    }
}

void uTCNAudioProcessorEditor::resized()
{
    auto marginTop          = 32;
    auto contentPadding     = 12;
    auto sectionPadding     = 18;
    auto contentItemHeight  = 24;
    auto rotaryItemHeight   = 55;
    auto stripWidth         = 30;
    auto sidePanelWidth     = 200;

    // side panel
    auto area = getLocalBounds();

    area.removeFromTop(marginTop + 60);
    area.removeFromLeft(600 - sidePanelWidth);
    area.removeFromLeft(sectionPadding+30);
    area.removeFromRight(sectionPadding+10);

    // place the gain sliders
    inputGainSlider.setBounds  (area.removeFromTop (rotaryItemHeight));
    area.removeFromTop(6);
    outputGainSlider.setBounds (area.removeFromTop (rotaryItemHeight));
    
    area.removeFromTop(12);
    area.removeFromLeft(65); // slide over the textboxes
    //area.removeFromRight(); // slide over the textboxes
    receptiveFieldTextEditor.setBounds(area.removeFromTop (contentItemHeight));
    area.removeFromTop(1);
    parametersTextEditor.setBounds(area.removeFromTop (contentItemHeight));
    area.removeFromTop(1);
    seedTextEditor.setBounds (area.removeFromTop (contentItemHeight));
    area.removeFromTop(1);

    // center panel
    area = getLocalBounds();

    area.removeFromTop(marginTop);
    area.removeFromLeft(stripWidth);
    area.removeFromLeft(sectionPadding+65);
    area.removeFromRight(sidePanelWidth);
    area.removeFromRight(sectionPadding);

    layersSlider.setBounds        (area.removeFromTop (contentItemHeight));
    area.removeFromTop(contentPadding);
    kernelSlider.setBounds        (area.removeFromTop (contentItemHeight));
    area.removeFromTop(contentPadding);
    channelsSlider.setBounds      (area.removeFromTop (contentItemHeight));
    area.removeFromTop(contentPadding);
    dilationsComboBox.setBounds   (area.removeFromTop (contentItemHeight));
    area.removeFromTop(contentPadding);
    activationsComboBox.setBounds (area.removeFromTop (contentItemHeight));
    area.removeFromTop(contentPadding);
    initTypeComboBox.setBounds    (area.removeFromTop (contentItemHeight));
    area.removeFromTop(contentPadding);

    auto toggleArea = area.removeFromTop (contentItemHeight);
    useBiasButton.setBounds       (toggleArea);
    linkGainButton.setBounds      (toggleArea.removeFromRight(60));
    depthwiseButton.setBounds     (toggleArea.removeFromRight(120));
}

