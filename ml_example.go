package main

import (
	"fmt"

	"github.com/cugbtang/tclaude/ml"
)

func main() {
	// Example 1: Vector operations
	fmt.Println("=== Vector Operations ===")
	
	// Create vectors
	v1 := ml.NewVector([]float64{1, 2, 3})
	v2 := ml.NewVector([]float64{4, 5, 6})
	
	fmt.Printf("Vector v1: %v\n", v1)
	fmt.Printf("Vector v2: %v\n", v2)
	
	// Vector addition
	v3 := v1.Add(v2)
	fmt.Printf("v1 + v2 = %v\n", v3)
	
	// Vector subtraction
	v4 := v1.Subtract(v2)
	fmt.Printf("v1 - v2 = %v\n", v4)
	
	// Dot product
	dotProduct := v1.DotProduct(v2)
	fmt.Printf("v1 · v2 = %.2f\n", dotProduct)
	
	// Magnitude
	magnitude := v1.Magnitude()
	fmt.Printf("|v1| = %.2f\n", magnitude)
	
	// Scaling
	v5 := v1.Scale(2)
	fmt.Printf("2 * v1 = %v\n", v5)
	
	fmt.Println()
	
	// Example 2: Matrix operations
	fmt.Println("=== Matrix Operations ===")
	
	// Create matrices
	m1 := ml.NewMatrix(2, 3)
	m1[0][0] = 1
	m1[0][1] = 2
	m1[0][2] = 3
	m1[1][0] = 4
	m1[1][1] = 5
	m1[1][2] = 6
	
	m2 := ml.NewMatrix(3, 2)
	m2[0][0] = 1
	m2[0][1] = 2
	m2[1][0] = 3
	m2[1][1] = 4
	m2[2][0] = 5
	m2[2][1] = 6
	
	fmt.Printf("Matrix m1:\n%v\n", m1)
	fmt.Printf("Matrix m2:\n%v\n", m2)
	
	// Matrix multiplication
	m3 := m1.Multiply(m2)
	fmt.Printf("m1 * m2 =\n%v\n", m3)
	
	// Matrix scaling
	m4 := m1.Scale(2)
	fmt.Printf("2 * m1 =\n%v\n", m4)
	
	// Matrix transpose
	m5 := m1.Transpose()
	fmt.Printf("m1^T =\n%v\n", m5)
	
	fmt.Println()
	
	// Example 3: Machine Learning Concepts
	fmt.Println("=== Machine Learning Concepts ===")
	
	// Feature vector example
	features := ml.NewVector([]float64{1.5, 2.3, 0.8}) // e.g., [height, weight, age]
	weights := ml.NewVector([]float64{0.5, -0.2, 0.1})  // model weights
	
	// Linear combination (basic prediction)
	prediction := features.DotProduct(weights)
	fmt.Printf("Feature vector: %v\n", features)
	fmt.Printf("Weight vector: %v\n", weights)
	fmt.Printf("Prediction (feature · weights): %.2f\n", prediction)
	
	// Gradient calculation example
	// For a simple linear function f(x) = wx + b, gradient is just the feature value
	gradient := features.Scale(0.1) // learning rate = 0.1
	fmt.Printf("Gradient (with learning rate 0.1): %v\n", gradient)
	
	// Parameter update (gradient descent step)
	newWeights := weights.Subtract(gradient)
	fmt.Printf("Updated weights: %v\n", newWeights)
}