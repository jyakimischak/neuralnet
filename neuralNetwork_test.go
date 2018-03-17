package neuralnet

import (
	"errors"
	"testing"
)

func TestNewNeuron(t *testing.T) {
	const nInputs = 10

	n, err := newNeuron(nInputs)
	if err != nil {
		t.Error("For numInput=", nInputs, ", error ", err)
	}
	if n.NumInputs != nInputs {
		t.Error(
			"For n.NumInputs",
			"Expected", nInputs,
			"Got", n.NumInputs,
		)
	}
	if len(n.Weights) != nInputs+1 {
		t.Error(
			"For len(n.Weights)",
			"Expected", nInputs+1,
			"Got", len(n.Weights),
		)
	}
	if len(n.Inputs) != nInputs+1 {
		t.Error(
			"For len(n.Inputs)",
			"Expected", nInputs+1,
			"Got", len(n.Inputs),
		)
	}

	_, err2 := newNeuron(0)
	if err2 == nil {
		t.Error("For numInput=0, did not recieve error")
	}

	_, err3 := newNeuron(-1)
	if err3 == nil {
		t.Error("For numInput=-1, did not recieve error")
	}
}

// getNeuronKnownState will give a neuron with a known state for testing purposes.
//output before activation:
//1 * 10 + 2 * 20 + 3 * 30 + 5 = 145
func getNeuronKnownState() (*neuron, error) {
	n, err := newNeuron(3)
	if err != nil {
		return n, errors.New("Error while creating known state neuron")
	}
	//set the inputs, weights and bias to known values
	n.Inputs[0] = 1
	n.Inputs[1] = 2
	n.Inputs[2] = 3
	n.Weights[0] = 10
	n.Weights[1] = 20
	n.Weights[2] = 30
	n.Bias = 5

	return n, nil
}

func TestCalc(t *testing.T) {
	nInvalid := neuron{}
	err := nInvalid.calc()
	if err == nil {
		t.Error("For invalid neuron, did not recieve error")
	}

	n, err2 := getNeuronKnownState()
	if err2 != nil {
		t.Error(err2)
	}

	err3 := n.calc()
	if err3 != nil {
		t.Error("Error while calling calc on known state neuron")
	}
	if n.OutBeforeAct != 145 {
		t.Error(
			"For n.OutBeforeAct",
			"Expected", 145,
			"Got", n.OutBeforeAct,
		)
	}

	//TODO test the activation functions

}
