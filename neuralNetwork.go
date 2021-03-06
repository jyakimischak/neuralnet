/*
Package neuralnet implements a "toy" neural net implementation.

author: Jonas Yakimischak
*/
package neuralnet

import (
	"errors"
	"fmt"
	"math/rand"
	"time"

	"github.com/jyakimischak/neuralnet/actfuncs"
)

const initialBias = 0

const layerTypeInput = "layerTypeInput"
const layerTypeHidden = "layerTypeHidden"
const layerTypeOutput = "layerTypeOutput"

const maxRecurseDepth = 30

//*************************************************************************************************************
//neuron

// neuron is a single neuron in a layer.
type neuron struct {
	Weights      []float64
	Inputs       []float64
	Output       float64
	OutBeforeAct float64
	NumInputs    int
	Bias         float64
	ActFunc      string
}

// newNeuron will setup a neuron and return the instance of it.
func newNeuron(numInputs int, actFunc string) (*neuron, error) {
	rand.Seed(time.Now().UTC().UnixNano())

	n := new(neuron)
	if numInputs < 1 {
		return n, errors.New("initNewNeuron: numInputs must be a positive integer")
	}

	n.NumInputs = numInputs

	//numInputs+1 so that the bias is included
	for i := 0; i < numInputs+1; i++ {
		//random init values
		n.Weights = append(n.Weights, rand.Float64())
		//placeholders
		n.Inputs = append(n.Inputs, 0)
	}
	n.Bias = initialBias

	n.ActFunc = actFunc

	return n, nil
}

// isNeuronValid will check if a neuron is in a valid state.
func (n *neuron) isValid() (bool, string) {
	if n.NumInputs < 1 {
		return false, fmt.Sprintf("Invalid neuron. NumInput must be a positive integer but is %d. Did you call newNeuron when getting the instance?", n.NumInputs)
	}
	if len(n.Weights) != n.NumInputs+1 || len(n.Inputs) != n.NumInputs+1 {
		return false, fmt.Sprintf("Invalid neuron. Weights/Input not initialized properly. Did you call newNeuron when getting the instance?")
	}
	return true, ""
}

// calc will calculate the output value for the neuron using the given activation function.
func (n *neuron) calc() error {
	isValid, isValidMsg := n.isValid()
	if !isValid {
		return errors.New(isValidMsg)
	}

	n.OutBeforeAct = 0
	for i := 0; i <= n.NumInputs; i++ {
		if i == n.NumInputs {
			n.OutBeforeAct += n.Bias
		} else {
			n.OutBeforeAct += n.Inputs[i] * n.Weights[i]
		}
	}

	n.Output = actfuncs.ApplyActFunc(n.ActFunc, n.OutBeforeAct)

	return nil
}

//*************************************************************************************************************
//neuralLayer

// NeuralLayer represents a layer in the neural network.
type neuralLayer struct {
	LayerType  string
	NumNeurons int
	Neurons    []*neuron
	PrevLayer  *neuralLayer
	NextLayer  *neuralLayer
	Inputs     []float64
	NumInputs  int
	Outputs    []float64
	ActFunc    string
}

// isValidLayerType will return true if the layer type is valid
func isValidLayerType(layerType string) bool {
	return layerType == layerTypeHidden || layerType == layerTypeInput || layerType == layerTypeOutput
}

// newNeuralLayer will setup a neural layer and return an instance of it.
// PrevLayer and NextLayer are NOT setup, they must be set after receiving the instance.
func newNeuralLayer(layerType string, numNeurons int, numInputs int, actFunc string) (*neuralLayer, error) {
	nl := &neuralLayer{}

	if !isValidLayerType(layerType) {
		return nl, fmt.Errorf("Unknown layer type: %s", layerType)
	}
	if numNeurons < 1 {
		return nl, fmt.Errorf("numNeurons must be greater than 1, it is: %d", numNeurons)
	}
	if numInputs < 1 {
		return nl, fmt.Errorf("numInputs must be greater than 1, it is: %d", numInputs)
	}
	if !actfuncs.IsValidActFunc(actFunc) {
		return nl, fmt.Errorf("Unknown activation function: %s", actFunc)
	}

	nl.LayerType = layerType
	if layerType == layerTypeInput {
		nl.ActFunc = actfuncs.NoActFunc
	}

	nl.NumNeurons = numNeurons
	for i := 0; i < numNeurons; i++ {
		n, err := newNeuron(numInputs, actFunc)
		if err != nil {
			return nl, err
		}
		nl.Neurons = append(nl.Neurons, n)
		nl.Outputs = append(nl.Outputs, 0)
	}

	nl.NumInputs = numInputs
	for i := 0; i < numInputs; i++ {
		nl.Inputs = append(nl.Inputs, 0)
	}

	nl.ActFunc = actFunc

	return nl, nil
}

func (nl *neuralLayer) isValid() (bool, string) {
	if !isValidLayerType(nl.LayerType) {
		return false, fmt.Sprintf("Invalid layer type: %s", nl.LayerType)
	}
	if nl.NumNeurons < 1 {
		return false, fmt.Sprintf("NumNeurons must be > 0 but is: %d", nl.NumNeurons)
	}
	if len(nl.Neurons) != nl.NumNeurons {
		return false, fmt.Sprintf("len(nl.Neurons) != nl.NumNeurons: %d, %d", len(nl.Neurons), nl.NumNeurons)
	}
	if len(nl.Outputs) != nl.NumNeurons {
		return false, fmt.Sprintf("len(nl.Outputs) != nl.NumNeurons: %d, %d", len(nl.Outputs), nl.NumNeurons)
	}
	if nl.NumInputs < 1 {
		return false, fmt.Sprintf("NumInputs must be > 0 but is: %d", nl.NumInputs)
	}
	if len(nl.Inputs) != nl.NumInputs {
		return false, fmt.Sprintf("len(nl.Inputs) != nl.NumInputs: %d, %d", len(nl.Inputs), nl.NumInputs)
	}
	if !actfuncs.IsValidActFunc(nl.ActFunc) {
		return false, fmt.Sprintf("Invalid activation function : %s", nl.ActFunc)
	}

	for iNeuron := 0; iNeuron < len(nl.Neurons); iNeuron++ {
		isValid, validMsg := nl.Neurons[iNeuron].isValid()
		if !isValid {
			return false, validMsg
		}
	}

	return true, ""
}

// calc will calculate the outputs for this neural layer.
func (nl *neuralLayer) calc() error {
	isValid, isValidMsg := nl.isValid()
	if !isValid {
		return errors.New(isValidMsg)
	}

	for iNeurons := 0; iNeurons < nl.NumNeurons; iNeurons++ {
		//load the inputs into the neuron
		for iInputs := 0; iInputs < nl.NumInputs; iInputs++ {
			nl.Neurons[iNeurons].Inputs[iInputs] = nl.Inputs[iInputs]
		}
		err := nl.Neurons[iNeurons].calc()
		if err != nil {
			return err
		}
		nl.Outputs[iNeurons] = nl.Neurons[iNeurons].Output
	}

	return nil
}

//*************************************************************************************************************
//NeuralNetwork

// NeuralNetwork is a pass forward neural network.
type NeuralNetwork struct {
	InputLayer   *neuralLayer
	HiddenLayers []*neuralLayer
	OutputLayer  *neuralLayer
}

// InputLayerProps is used when calling NewNeuralNetwork.
type InputLayerProps struct {
	NumInputs int
}

// HiddenLayerProps is used when calling NewNeuralNetwork.
type HiddenLayerProps struct {
	NumNeurons int
	ActFunc    string
}

// OutputLayerProps is used when calling NewNeuralNetwork.
type OutputLayerProps struct {
	NumOutputs int
	ActFunc    string
}

// NewNeuralNetwork get an instance of a netral network.
func NewNeuralNetwork(inputLayerProps InputLayerProps, hiddenLayerProps []HiddenLayerProps, outputLayerProps OutputLayerProps) (*NeuralNetwork, error) {
	nn := &NeuralNetwork{}

	//validate
	if inputLayerProps.NumInputs < 1 {
		return nn, fmt.Errorf("inputLayerProps.NumInputs must be > 0 and is: %d", inputLayerProps.NumInputs)
	}
	for iHiddenLayer := 0; iHiddenLayer < len(hiddenLayerProps); iHiddenLayer++ {
		if hiddenLayerProps[iHiddenLayer].NumNeurons < 1 {
			return nn, fmt.Errorf("hiddenLayerProps[%d].NumNeurons must be > 0 and is: %d", iHiddenLayer, hiddenLayerProps[iHiddenLayer].NumNeurons)
		}
		if !actfuncs.IsValidActFunc(hiddenLayerProps[iHiddenLayer].ActFunc) {
			return nn, fmt.Errorf("hiddenLayerProps[%d].ActFunc is unknown: %s", iHiddenLayer, hiddenLayerProps[iHiddenLayer].ActFunc)
		}
	}
	if outputLayerProps.NumOutputs < 1 {
		return nn, fmt.Errorf("outputLayerProps.NumOutputs must be > 0 and is: %d", outputLayerProps.NumOutputs)
	}
	if !actfuncs.IsValidActFunc(outputLayerProps.ActFunc) {
		return nn, fmt.Errorf("outputLayerProps.ActFunc is unknown: %s", outputLayerProps.ActFunc)
	}

	//create the input layer
	il, err := newNeuralLayer(layerTypeInput, inputLayerProps.NumInputs, inputLayerProps.NumInputs, actfuncs.NoActFunc)
	if err != nil {
		return nn, err
	}
	nn.InputLayer = il

	//create the hidden layers
	for iHiddenLayer := 0; iHiddenLayer < len(hiddenLayerProps); iHiddenLayer++ {
		var hl *neuralLayer
		var err error
		if iHiddenLayer == 0 {
			hl, err = newNeuralLayer(layerTypeHidden, hiddenLayerProps[iHiddenLayer].NumNeurons, inputLayerProps.NumInputs, hiddenLayerProps[iHiddenLayer].ActFunc)
			if err != nil {
				return nn, err
			}
			nn.InputLayer.NextLayer = hl
			hl.PrevLayer = nn.InputLayer
		} else {
			hl, err = newNeuralLayer(layerTypeHidden, hiddenLayerProps[iHiddenLayer].NumNeurons, hiddenLayerProps[iHiddenLayer-1].NumNeurons, hiddenLayerProps[iHiddenLayer].ActFunc)
			if err != nil {
				return nn, err
			}
			nn.HiddenLayers[iHiddenLayer-1].NextLayer = hl
			hl.PrevLayer = nn.HiddenLayers[iHiddenLayer-1]
		}
		nn.HiddenLayers = append(nn.HiddenLayers, hl)
	}

	//create the output layer
	var ol *neuralLayer
	var err2 error
	if len(hiddenLayerProps) > 0 {
		ol, err2 = newNeuralLayer(layerTypeOutput, outputLayerProps.NumOutputs, hiddenLayerProps[len(hiddenLayerProps)-1].NumNeurons, outputLayerProps.ActFunc)
		if err2 != nil {
			return nn, err2
		}
		nn.HiddenLayers[len(hiddenLayerProps)-1].NextLayer = ol
		ol.PrevLayer = nn.HiddenLayers[len(hiddenLayerProps)-1]
	} else {
		ol, err2 = newNeuralLayer(layerTypeOutput, outputLayerProps.NumOutputs, inputLayerProps.NumInputs, outputLayerProps.ActFunc)
		if err2 != nil {
			return nn, err2
		}
		nn.InputLayer.NextLayer = ol
		ol.PrevLayer = nn.InputLayer
	}
	nn.OutputLayer = ol

	return nn, nil
}

// IsValid checks if this neural network is in a valid state.
func (nn *NeuralNetwork) IsValid() (bool, string) {
	return nn.isValidRecurse(1, nil, nn.InputLayer)
}
func (nn *NeuralNetwork) isValidRecurse(depth int, prevLayer *neuralLayer, layer *neuralLayer) (bool, string) {
	if depth == maxRecurseDepth {
		return false, "Max recurse depth reached while validating"
	}
	var isValid bool
	var invalidMsg string
	isValid, invalidMsg = layer.isValid()
	if !isValid {
		return false, invalidMsg
	}
	if layer.NumInputs != len(layer.Inputs) {
		return false, fmt.Sprintf("At depth %d, layer.NumInputs != len(layer.Inputs): %d, %d", depth, layer.NumInputs, len(layer.Inputs))
	}
	if layer.NumNeurons != len(layer.Neurons) {
		return false, fmt.Sprintf("At depth %d, layer.NumNeurons != len(layer.Neurons): %d, %d", depth, layer.NumNeurons, len(layer.Neurons))
	}
	if prevLayer != layer.PrevLayer {
		return false, fmt.Sprintf("At depth %d, prevLayer != layer.PrevLayer", depth)
	}

	//if we reached the output layer then we're all good
	if layer.LayerType == layerTypeOutput {
		return true, ""
	}

	if layer.NextLayer == nil {
		return false, "Nil next layer found before the output layer"
	}

	if len(layer.Outputs) != layer.NextLayer.NumInputs {
		return false, fmt.Sprintf("At depth %d, len(layer.Outputs) != layer.NextLayer.NumInputs: %d, %d", depth, len(layer.Outputs), layer.NextLayer.NumInputs)
	}

	return nn.isValidRecurse(depth+1, layer, layer.NextLayer)
}

//Calc will run the inputs through all layers of the neural network and save the outputs.
func (nn *NeuralNetwork) Calc() error {
	isValid, invalidMsg := nn.IsValid()
	if !isValid {
		return errors.New(invalidMsg)
	}
	return nn.calcRecurse(1, nn.InputLayer)
}
func (nn *NeuralNetwork) calcRecurse(depth int, layer *neuralLayer) error {
	if depth == maxRecurseDepth {
		return errors.New("Max recurse depth reached")
	}
	layer.calc()
	//if we hit the output layer we are done
	if layer.LayerType == layerTypeOutput {
		return nil
	}
	copy(layer.NextLayer.Inputs, layer.Outputs)
	return nn.calcRecurse(depth+1, layer.NextLayer)
}
