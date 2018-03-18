package neuralnet

import (
	"errors"
	"testing"

	"github.com/jyakimischak/neuralnet/actfuncs"
)

func TestNewNeuron(t *testing.T) {
	const nInputs = 10

	n, err := newNeuron(nInputs, actfuncs.NoActFunc)
	if err != nil {
		t.Error("For numInput=", nInputs, ", error ", err)
	}
	if n.NumInputs != nInputs {
		t.Error("For n.NumInputs", "Expected", nInputs, "Got", n.NumInputs)
	}
	if len(n.Weights) != nInputs+1 {
		t.Error("For len(n.Weights)", "Expected", nInputs+1, "Got", len(n.Weights))
	}
	if len(n.Inputs) != nInputs+1 {
		t.Error("For len(n.Inputs)", "Expected", nInputs+1, "Got", len(n.Inputs))
	}

	_, err2 := newNeuron(0, actfuncs.NoActFunc)
	if err2 == nil {
		t.Error("For numInput=0, did not recieve error")
	}

	_, err3 := newNeuron(-1, actfuncs.NoActFunc)
	if err3 == nil {
		t.Error("For numInput=-1, did not recieve error")
	}
}

// getNeuronKnownState will give a neuron with a known state for testing purposes.
//output before activation:
//1 * 10 + 2 * 20 + 3 * 30 + 5 = 145
func getNeuronKnownState() (*neuron, error) {
	n, err := newNeuron(3, actfuncs.NoActFunc)
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

func TestNeuronCalc(t *testing.T) {
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
		t.Error("For n.OutBeforeAct", "Expected", 145, "Got", n.OutBeforeAct)
	}
	if n.Output != 145 {
		t.Error("For n.Output no activation function", "Expected", 145, "Got", n.Output)
	}

	n.ActFunc = actfuncs.Step
	err4 := n.calc()
	if err4 != nil {
		t.Error("Error while calling calc on known state neuron for step activation function")
	}
	if n.Output != 1 {
		t.Error("For n.Output Step", "Expected", 1, "Got", n.Output)
	}

	n.ActFunc = actfuncs.Sigmoid
	err5 := n.calc()
	if err5 != nil {
		t.Error("Error while calling calc on known state neuron for sigmoid activation function")
	}
	if n.Output != 1 {
		t.Error("For n.Output Step", "Expected", 1, "Got", n.Output)
	}
}

func TestNewNeuralLayer(t *testing.T) {
	_, err := newNeuralLayer("invalid", 10, 5, actfuncs.NoActFunc)
	if err == nil {
		t.Error("For invalid layer type, did not recieve error")
	}

	_, err2 := newNeuralLayer(layerTypeInput, 0, 5, actfuncs.NoActFunc)
	if err2 == nil {
		t.Error("For num neutrons 0, did not recieve error")
	}

	_, err3 := newNeuralLayer(layerTypeInput, 10, 0, actfuncs.NoActFunc)
	if err3 == nil {
		t.Error("For num inputs 0, did not recieve error")
	}

	_, err4 := newNeuralLayer(layerTypeInput, 10, 5, "invalid")
	if err4 == nil {
		t.Error("For invalid activation function, did not recieve error")
	}

	nl, err5 := newNeuralLayer(layerTypeInput, 10, 5, actfuncs.NoActFunc)
	if err5 != nil {
		t.Error("For valid neural layer, recieved error")
	}
	if nl.LayerType != layerTypeInput {
		t.Error("For nl.LayerType", "Expected", layerTypeInput, "Got", nl.LayerType)
	}
	if nl.NumNeurons != 10 {
		t.Error("For nl.NumNeurons", "Expected", 10, "Got", nl.NumNeurons)
	}
	if len(nl.Neurons) != 10 {
		t.Error("For len(nl.Neurons)", "Expected", 10, "Got", nl.Neurons)
	}
	if len(nl.Outputs) != 10 {
		t.Error("For len(nl.Outputs)", "Expected", 10, "Got", nl.Outputs)
	}
	if nl.NumInputs != 5 {
		t.Error("For nl.NumInputs", "Expected", 5, "Got", nl.NumInputs)
	}
	if len(nl.Inputs) != 5 {
		t.Error("For len(nl.Inputs)", "Expected", 5, "Got", nl.NumInputs)
	}
	if nl.ActFunc != actfuncs.NoActFunc {
		t.Error("For nl.ActFunc", "Expected", actfuncs.NoActFunc, "Got", nl.ActFunc)
	}

	//need to check a non-input layer to ensure the activation function is set properly
	nl2, err6 := newNeuralLayer(layerTypeHidden, 10, 5, actfuncs.Sigmoid)
	if err6 != nil {
		t.Error("For valid neural layer, recieved error")
	}
	if nl2.ActFunc != actfuncs.Sigmoid {
		t.Error("For nl2.ActFunc", "Expected", actfuncs.Sigmoid, "Got", nl2.ActFunc)
	}

}

// getKnownStateNeuralLayer will return a neural layer with known state.
// This uses getKnownStateNeuron for its neurons.
func getKnownStateNeuralLayer() (*neuralLayer, error) {
	nl, err := newNeuralLayer(layerTypeInput, 3, 3, actfuncs.NoActFunc)
	if err != nil {
		return nl, err
	}
	for iNeurons := 0; iNeurons < 3; iNeurons++ {
		nl.Neurons[iNeurons], err = getNeuronKnownState()
		if err != nil {
			return nl, err
		}
	}
	nl.Inputs[0] = 1
	nl.Inputs[1] = 2
	nl.Inputs[2] = 3

	return nl, nil
}

func TestNeuralLayerCalc(t *testing.T) {
	nlInvalid := neuralLayer{}
	err := nlInvalid.calc()
	if err == nil {
		t.Error("For invalid neuralLayer, did not recieve error")
	}

	nl, err2 := getKnownStateNeuralLayer()
	if err2 != nil {
		t.Error("Error getting known state neural layer")
	}
	err3 := nl.calc()
	if err3 != nil {
		t.Error(err3)
	}

	for iInputs := 0; iInputs < 3; iInputs++ {
		if nl.Outputs[iInputs] != 145 {
			t.Errorf("For nl.Outputs[%d] Expected 145 Got %f", iInputs, nl.Outputs[iInputs])
		}
	}
}

func TestNewNeuralNetwork(t *testing.T) {
	_, err := NewNeuralNetwork(
		InputLayerProps{NumInputs: 0},
		nil,
		OutputLayerProps{NumOutputs: 1, ActFunc: actfuncs.Sigmoid},
	)
	if err == nil {
		t.Error("For InputLayerProps{NumInputs: 0}, did not recieve error")
	}

	_, err2 := NewNeuralNetwork(
		InputLayerProps{NumInputs: 1},
		[]HiddenLayerProps{
			HiddenLayerProps{NumNeurons: 0, ActFunc: actfuncs.Sigmoid},
		},
		OutputLayerProps{NumOutputs: 1, ActFunc: actfuncs.Sigmoid},
	)
	if err2 == nil {
		t.Error("For HiddenLayerProps{NumNeurons: 0, ActFunc: actfuncs.Sigmoid}, did not recieve error")
	}

	_, err3 := NewNeuralNetwork(
		InputLayerProps{NumInputs: 1},
		[]HiddenLayerProps{
			HiddenLayerProps{NumNeurons: 1, ActFunc: "invalid"},
		},
		OutputLayerProps{NumOutputs: 1, ActFunc: actfuncs.Sigmoid},
	)
	if err3 == nil {
		t.Error("For HiddenLayerProps{NumNeurons: 1, ActFunc: \"invalid\"}, did not recieve error")
	}

	_, err4 := NewNeuralNetwork(
		InputLayerProps{NumInputs: 1},
		nil,
		OutputLayerProps{NumOutputs: 0, ActFunc: actfuncs.Sigmoid},
	)
	if err4 == nil {
		t.Error("For OutputLayerProps{NumOutputs: 0, ActFunc: actfuncs.Sigmoid}, did not recieve error")
	}

	_, err5 := NewNeuralNetwork(
		InputLayerProps{NumInputs: 1},
		nil,
		OutputLayerProps{NumOutputs: 1, ActFunc: "invalid"},
	)
	if err5 == nil {
		t.Error("For OutputLayerProps{NumOutputs: 1, ActFunc: \"invalid\"}, did not recieve error")
	}

	nn, err6 := NewNeuralNetwork(
		InputLayerProps{NumInputs: 1},
		[]HiddenLayerProps{
			HiddenLayerProps{NumNeurons: 10, ActFunc: actfuncs.Sigmoid},
			HiddenLayerProps{NumNeurons: 20, ActFunc: actfuncs.Step},
		},
		OutputLayerProps{NumOutputs: 2, ActFunc: actfuncs.Sigmoid},
	)
	if err6 != nil {
		t.Error(err6)
	}

	if len(nn.InputLayer.Neurons) != 1 {
		t.Error("For len(nn.InputLayer.Neurons)", "Expected", 1, "Got", len(nn.InputLayer.Neurons))
	}
	if nn.InputLayer.ActFunc != actfuncs.NoActFunc {
		t.Error("For nn.InputLayer.ActFunc", "Expected", actfuncs.NoActFunc, "Got", nn.InputLayer.ActFunc)
	}

	if len(nn.HiddenLayers) != 2 {
		t.Error("For len(nn.HiddenLayers)", "Expected", 2, "Got", len(nn.HiddenLayers))
	}
	if len(nn.HiddenLayers[0].Inputs) != 1 {
		t.Error("For len(nn.HiddenLayers[0].Inputs)", "Expected", 1, "Got", len(nn.HiddenLayers[0].Inputs))
	}
	if len(nn.HiddenLayers[0].Neurons) != 10 {
		t.Error("For len(nn.HiddenLayers[0].Neurons)", "Expected", 10, "Got", len(nn.HiddenLayers[0].Neurons))
	}
	if nn.HiddenLayers[0].ActFunc != actfuncs.Sigmoid {
		t.Error("For nn.HiddenLayers[0].ActFunc", "Expected", actfuncs.Sigmoid, "Got", nn.HiddenLayers[0].ActFunc)
	}
	if len(nn.HiddenLayers[1].Inputs) != 10 {
		t.Error("For len(nn.HiddenLayers[1].Inputs)", "Expected", 10, "Got", len(nn.HiddenLayers[1].Inputs))
	}
	if len(nn.HiddenLayers[1].Neurons) != 20 {
		t.Error("For len(nn.HiddenLayers[1].Neurons)", "Expected", 20, "Got", len(nn.HiddenLayers[1].Neurons))
	}
	if nn.HiddenLayers[1].ActFunc != actfuncs.Step {
		t.Error("For nn.HiddenLayers[1].ActFunc", "Expected", actfuncs.Step, "Got", nn.HiddenLayers[1].ActFunc)
	}

	if len(nn.OutputLayer.Inputs) != 20 {
		t.Error("For len(nn.OutputLayer.Inputs)", "Expected", 20, "Got", len(nn.OutputLayer.Inputs))
	}
	if len(nn.OutputLayer.Neurons) != 2 {
		t.Error("For len(nn.OutputLayer.Neurons)", "Expected", 2, "Got", len(nn.OutputLayer.Neurons))
	}
	if nn.OutputLayer.ActFunc != actfuncs.Sigmoid {
		t.Error("For nn.OutputLayer.ActFunc", "Expected", actfuncs.Sigmoid, "Got", nn.OutputLayer.ActFunc)
	}

	//forward traverse
	if nn.InputLayer.NextLayer.NextLayer.NextLayer.LayerType != layerTypeOutput {
		t.Error("For nn.InputLayer.NextLayer.NextLayer.NextLayer.LayerType", "Expected", layerTypeOutput, "Got", nn.InputLayer.NextLayer.NextLayer.NextLayer.LayerType)
	}
	//backward traverse
	if nn.OutputLayer.PrevLayer.PrevLayer.PrevLayer.LayerType != layerTypeInput {
		t.Error("For nn.OutputLayer.PrevLayer.PrevLayer.PrevLayer.LayerType != layerTypeInput", "Expected", layerTypeInput, "Got", nn.OutputLayer.PrevLayer.PrevLayer.PrevLayer.LayerType)
	}

	nn2, err7 := NewNeuralNetwork(
		InputLayerProps{NumInputs: 1},
		nil,
		OutputLayerProps{NumOutputs: 2, ActFunc: actfuncs.Sigmoid},
	)
	if err7 != nil {
		t.Error(err7)
	}
	//forward traverse
	if nn2.InputLayer.NextLayer.LayerType != layerTypeOutput {
		t.Error("For nn2.InputLayer.NextLayer.LayerType", "Expected", layerTypeOutput, "Got", nn2.InputLayer.NextLayer.LayerType)
	}
	//backward traverse
	if nn2.OutputLayer.PrevLayer.LayerType != layerTypeInput {
		t.Error("For nn2.OutputLayer.PrevLayer.LayerType != layerTypeInput", "Expected", layerTypeInput, "Got", nn2.OutputLayer.PrevLayer.LayerType)
	}
}

func TestNauralNetworkCalc(t *testing.T) {
	nn, err := NewNeuralNetwork(
		InputLayerProps{NumInputs: 3},
		[]HiddenLayerProps{
			HiddenLayerProps{NumNeurons: 10, ActFunc: actfuncs.Sigmoid},
			HiddenLayerProps{NumNeurons: 20, ActFunc: actfuncs.Sigmoid},
		},
		OutputLayerProps{NumOutputs: 2, ActFunc: actfuncs.NoActFunc},
	)
	if err != nil {
		t.Error(err)
	}

	nn.InputLayer.Inputs[0] = 0.003
	nn.InputLayer.Inputs[1] = 0.008
	nn.InputLayer.Inputs[2] = 0.002
	err2 := nn.Calc()
	if err2 != nil {
		t.Error(err2)
	}
	// fmt.Printf("\n\noutput %f, %f\n\n", nn.OutputLayer.Outputs[0], nn.OutputLayer.Outputs[1])

	// t.Error("bla")
}
