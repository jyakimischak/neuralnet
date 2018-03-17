package actfuncs

import (
	"math"
	"testing"
)

// sigmoidSolveForX will take a given x and return the y for the sigmoid function.
func sigmoidSolveForX(y float64) float64 {
	return -1 * math.Log((1-y)/y)
}

func TestCalcSigmoid(t *testing.T) {
	y1Val := 0.001
	y2Val := 0.1
	y3Val := 0.5
	y4Val := 0.6
	y5Val := 0.991

	y1 := calcSigmoid(sigmoidSolveForX(y1Val))
	if y1 != y1Val {
		t.Error("Expected", y1Val, "Got", y1)
	}
	y2 := calcSigmoid(sigmoidSolveForX(y2Val))
	if y2 != y2Val {
		t.Error("Expected", y2Val, "Got", y2)
	}
	y3 := calcSigmoid(sigmoidSolveForX(y3Val))
	if y3 != y3Val {
		t.Error("Expected", y3Val, "Got", y3)
	}
	y4 := calcSigmoid(sigmoidSolveForX(y4Val))
	if y4 != y4Val {
		t.Error("Expected", y4Val, "Got", y4)
	}
	y5 := calcSigmoid(sigmoidSolveForX(y5Val))
	if y5 != y5Val {
		t.Error("Expected", y5Val, "Got", y5)
	}
}

func TestCalcStep(t *testing.T) {
	x1 := -10.0
	x2 := -0.1
	x3 := 0.1
	x4 := 10.0

	y1 := calcStep(x1)
	if y1 != 0 {
		t.Error("For", x1, "Expected", 0, "Got", y1)
	}
	y2 := calcStep(x2)
	if y2 != 0 {
		t.Error("For", x2, "Expected", 0, "Got", y2)
	}
	y3 := calcStep(x3)
	if y3 != 1 {
		t.Error("For", x3, "Expected", 1, "Got", y3)
	}
	y4 := calcStep(x4)
	if y4 != 1 {
		t.Error("For", x4, "Expected", 1, "Got", y4)
	}

}

func TestApplyActFunc(t *testing.T) {
	ySigmoidVal := 0.001
	ySigmoid := ApplyActFunc(Sigmoid, sigmoidSolveForX(ySigmoidVal))
	if ySigmoid != ySigmoidVal {
		t.Error("For Sigmoid Expected", ySigmoidVal, "Got", ySigmoidVal)
	}

	xStep := -10.0
	yStep := ApplyActFunc(Step, xStep)
	if yStep != 0 {
		t.Error("For Step Expected", 0, "Got", yStep)
	}

	yNone := ApplyActFunc("", 10)
	if yNone != 10 {
		t.Error("For none Expected", 10, "Got", yNone)
	}

}
