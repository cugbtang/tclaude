// Package ml provides machine learning fundamentals including vector and matrix operations
package ml

import (
	"fmt"
	"math"
)

// Vector represents a mathematical vector
type Vector []float64

// Matrix represents a mathematical matrix
type Matrix [][]float64

// NewVector creates a new vector
func NewVector(elements []float64) Vector {
	vec := make(Vector, len(elements))
	copy(vec, elements)
	return vec
}

// NewMatrix creates a new matrix
func NewMatrix(rows, cols int) Matrix {
	matrix := make(Matrix, rows)
	for i := range matrix {
		matrix[i] = make([]float64, cols)
	}
	return matrix
}

// Add adds two vectors
func (v Vector) Add(other Vector) Vector {
	if len(v) != len(other) {
		panic("vectors must have the same length")
	}
	
	result := make(Vector, len(v))
	for i := range v {
		result[i] = v[i] + other[i]
	}
	return result
}

// Subtract subtracts two vectors
func (v Vector) Subtract(other Vector) Vector {
	if len(v) != len(other) {
		panic("vectors must have the same length")
	}
	
	result := make(Vector, len(v))
	for i := range v {
		result[i] = v[i] - other[i]
	}
	return result
}

// DotProduct calculates the dot product of two vectors
func (v Vector) DotProduct(other Vector) float64 {
	if len(v) != len(other) {
		panic("vectors must have the same length")
	}
	
	var result float64
	for i := range v {
		result += v[i] * other[i]
	}
	return result
}

// Magnitude calculates the magnitude (length) of a vector
func (v Vector) Magnitude() float64 {
	var sumSquares float64
	for _, val := range v {
		sumSquares += val * val
	}
	return math.Sqrt(sumSquares)
}

// Scale multiplies a vector by a scalar
func (v Vector) Scale(scalar float64) Vector {
	result := make(Vector, len(v))
	for i, val := range v {
		result[i] = val * scalar
	}
	return result
}

// String returns a string representation of a vector
func (v Vector) String() string {
	return fmt.Sprintf("%v", []float64(v))
}

// Add adds two matrices
func (m Matrix) Add(other Matrix) Matrix {
	if len(m) != len(other) || len(m[0]) != len(other[0]) {
		panic("matrices must have the same dimensions")
	}
	
	result := NewMatrix(len(m), len(m[0]))
	for i := range m {
		for j := range m[i] {
			result[i][j] = m[i][j] + other[i][j]
		}
	}
	return result
}

// Multiply multiplies two matrices
func (m Matrix) Multiply(other Matrix) Matrix {
	if len(m[0]) != len(other) {
		panic("number of columns in first matrix must equal number of rows in second matrix")
	}
	
	result := NewMatrix(len(m), len(other[0]))
	for i := range m {
		for j := range other[0] {
			for k := range other {
				result[i][j] += m[i][k] * other[k][j]
			}
		}
	}
	return result
}

// Scale multiplies a matrix by a scalar
func (m Matrix) Scale(scalar float64) Matrix {
	result := NewMatrix(len(m), len(m[0]))
	for i := range m {
		for j := range m[i] {
			result[i][j] = m[i][j] * scalar
		}
	}
	return result
}

// Transpose returns the transpose of a matrix
func (m Matrix) Transpose() Matrix {
	result := NewMatrix(len(m[0]), len(m))
	for i := range m {
		for j := range m[i] {
			result[j][i] = m[i][j]
		}
	}
	return result
}

// String returns a string representation of a matrix
func (m Matrix) String() string {
	result := "["
	for i, row := range m {
		if i > 0 {
			result += " "
		}
		result += fmt.Sprintf("%v", row)
		if i < len(m)-1 {
			result += "\n"
		}
	}
	result += "]"
	return result
}