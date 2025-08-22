package ml

import (
	"testing"
)

func TestVectorOperations(t *testing.T) {
	// Test NewVector
	v1 := NewVector([]float64{1, 2, 3})
	v2 := NewVector([]float64{4, 5, 6})

	// Test Add
	v3 := v1.Add(v2)
	expectedAdd := []float64{5, 7, 9}
	for i, val := range v3 {
		if val != expectedAdd[i] {
			t.Errorf("Vector Add failed. Expected %v, got %v", expectedAdd, []float64(v3))
		}
	}

	// Test Subtract
	v4 := v1.Subtract(v2)
	expectedSub := []float64{-3, -3, -3}
	for i, val := range v4 {
		if val != expectedSub[i] {
			t.Errorf("Vector Subtract failed. Expected %v, got %v", expectedSub, []float64(v4))
		}
	}

	// Test DotProduct
	dotProduct := v1.DotProduct(v2)
	expectedDot := 32.0
	if dotProduct != expectedDot {
		t.Errorf("Vector DotProduct failed. Expected %f, got %f", expectedDot, dotProduct)
	}

	// Test Magnitude
	magnitude := v1.Magnitude()
	expectedMag := 3.7416573867739413 // sqrt(14)
	if magnitude != expectedMag {
		t.Errorf("Vector Magnitude failed. Expected %f, got %f", expectedMag, magnitude)
	}

	// Test Scale
	v5 := v1.Scale(2)
	expectedScale := []float64{2, 4, 6}
	for i, val := range v5 {
		if val != expectedScale[i] {
			t.Errorf("Vector Scale failed. Expected %v, got %v", expectedScale, []float64(v5))
		}
	}
}

func TestMatrixOperations(t *testing.T) {
	// Test NewMatrix
	m1 := NewMatrix(2, 3)
	m1[0][0] = 1
	m1[0][1] = 2
	m1[0][2] = 3
	m1[1][0] = 4
	m1[1][1] = 5
	m1[1][2] = 6

	m2 := NewMatrix(3, 2)
	m2[0][0] = 1
	m2[0][1] = 2
	m2[1][0] = 3
	m2[1][1] = 4
	m2[2][0] = 5
	m2[2][1] = 6

	// Test Add
	m3 := NewMatrix(2, 3)
	m3[0][0] = 2
	m3[0][1] = 4
	m3[0][2] = 6
	m3[1][0] = 8
	m3[1][1] = 10
	m3[1][2] = 12

	m4 := m1.Add(m3)
	expectedAdd := [][]float64{{3, 6, 9}, {12, 15, 18}}
	for i, row := range m4 {
		for j, val := range row {
			if val != expectedAdd[i][j] {
				t.Errorf("Matrix Add failed. Expected %v, got %v", expectedAdd, m4)
			}
		}
	}

	// Test Multiply
	m5 := m1.Multiply(m2)
	expectedMul := [][]float64{{22, 28}, {49, 64}}
	for i, row := range m5 {
		for j, val := range row {
			if val != expectedMul[i][j] {
				t.Errorf("Matrix Multiply failed. Expected %v, got %v", expectedMul, m5)
			}
		}
	}

	// Test Scale
	m6 := m1.Scale(2)
	expectedScale := [][]float64{{2, 4, 6}, {8, 10, 12}}
	for i, row := range m6 {
		for j, val := range row {
			if val != expectedScale[i][j] {
				t.Errorf("Matrix Scale failed. Expected %v, got %v", expectedScale, m6)
			}
		}
	}

	// Test Transpose
	m7 := m1.Transpose()
	expectedTrans := [][]float64{{1, 4}, {2, 5}, {3, 6}}
	for i, row := range m7 {
		for j, val := range row {
			if val != expectedTrans[i][j] {
				t.Errorf("Matrix Transpose failed. Expected %v, got %v", expectedTrans, m7)
			}
		}
	}
}

func TestVectorPanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Vector operations should panic with different lengths")
		}
	}()

	v1 := NewVector([]float64{1, 2, 3})
	v2 := NewVector([]float64{4, 5})
	v1.Add(v2)
}

func TestMatrixPanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Matrix operations should panic with incompatible dimensions")
		}
	}()

	m1 := NewMatrix(2, 3)
	m2 := NewMatrix(2, 3)
	m1.Multiply(m2)
}