/*
Package actfuncs defines the activation functions for the neurons.

author: Jonas Yakimischak
*/
package actfuncs

import "math"

//NoActFunc no activation function
const NoActFunc = "noActFunc"

//Step activation function
const Step = "step"

//Sigmoid activation function
const Sigmoid = "sigmoid"

// IsValidActFunc will return true if the given string is a valid activation function.
func IsValidActFunc(actFunc string) bool {
	return actFunc == NoActFunc || actFunc == Step || actFunc == Sigmoid
}

// ApplyActFunc will apply the given activation function and return the value.  If the activation function is unknown (or nil) then
// the x value will be returned without being modified.
func ApplyActFunc(actFunc string, x float64) float64 {
	switch actFunc {
	case Step:
		return calcStep(x)
	case Sigmoid:
		return calcSigmoid(x)
	default:
		return x
	}
}

// calcStep calculate step activation function
func calcStep(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

// calcSigmoid calculate sigmoid activation function
func calcSigmoid(x float64) float64 {
	return 1 / (1 + math.Pow(math.E, x*-1))
}
