/*
Package neuralnet implements a "toy" neural net implementation.

author: Jonas Yakimischak
*/
package neuralnet

import (
	"errors"
	"fmt"
	"math/rand"

	"github.com/jyakimischak/neuralnet/actfuncs"
)

const initialBias = 1.0

const layerTypeInput = "layerTypeInput"
const layerTypeHidden = "layerTypeHidden"
const layerTypeOutput = "layerTypeOutput"

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
	InputLayer   neuralLayer
	HiddenLayers []neuralLayer
	OutputLayer  neuralLayer
}

// func NewNeuralNetwork(inputLayerNumInputs int, outputLayerNumOutputs int, outputLayerActivationFunction string, )
