/*
Package neuralnet implements a "toy" neural net implementation.

author: Jonas Yakimischak
*/
package neuralnet

import (
	"errors"
	"fmt"
	"math/rand"
)

const initialBias = 1.0

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
}

// newNeuron will setup a neuron and return the instance of it.
func newNeuron(numInputs int) (*neuron, error) {
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

	return n, nil
}

// isNeuronValid will check if a neuron is in a valid state.
func (n *neuron) isNeuronValid() (bool, string) {
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
	isValid, isValidMsg := n.isNeuronValid()
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

	//TODO ensure that the activation function gets applied here
	n.Output = n.OutBeforeAct

	return nil
}

//*************************************************************************************************************
//neuralLayer

// NeuralLayer is blabla
type neuralLayer struct {
	NueronsInLayer int
	neurons        []*neuron
	PrevLayer      *neuralLayer
	NextLayer      *neuralLayer
	Input          []float64
	Output         []float64
	NumInputs      int
}
