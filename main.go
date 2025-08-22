package main

import (
	"fmt"
	"math"
)

// Point represents a 2D point
type Point struct {
	X, Y float64
}

// gradientDescent performs gradient descent to find the minimum of a function
// f(x) = (x - 2)^2 + 3
func gradientDescent(startX, learningRate float64, iterations int) float64 {
	x := startX

	for i := 0; i < iterations; i++ {
		// Calculate gradient: f'(x) = 2(x - 2)
		gradient := 2 * (x - 2)

		// Update x using gradient descent formula
		x = x - learningRate*gradient
	}

	return x
}

// linearRegression performs linear regression using gradient descent
func linearRegression(points []Point, learningRate float64, iterations int) (float64, float64) {
	// Initialize parameters
	m := 0.0 // slope
	b := 0.0 // y-intercept

	n := float64(len(points))

	for i := 0; i < iterations; i++ {
		// Calculate gradients
		dm := 0.0 // gradient for slope
		db := 0.0 // gradient for y-intercept

		for _, point := range points {
			// Predicted value
			yPred := m*point.X + b

			// Calculate errors
			error := yPred - point.Y

			// Accumulate gradients
			dm += (2 / n) * error * point.X
			db += (2 / n) * error
		}

		// Update parameters
		m = m - learningRate*dm
		b = b - learningRate*db
	}

	return m, b
}

func main() {
	// Example 1: Simple function minimization
	fmt.Println("Example 1: Minimizing f(x) = (x - 2)^2 + 3")
	startX := 0.0
	learningRate := 0.1
	iterations := 100

	minX := gradientDescent(startX, learningRate, iterations)
	minValue := math.Pow(minX-2, 2) + 3

	fmt.Printf("Minimum found at x = %.6f, f(x) = %.6f\n", minX, minValue)
	fmt.Println("Expected minimum at x = 2.000000, f(x) = 3.000000\n")

	// Example 2: Linear regression
	fmt.Println("Example 2: Linear regression")
	// Sample data points: y = 2x + 1 (with some noise)
	points := []Point{
		{1, 3.1},
		{2, 5.0},
		{3, 7.2},
		{4, 8.9},
		{5, 11.0},
	}

	learningRate = 0.01
	iterations = 1000

	m, b := linearRegression(points, learningRate, iterations)

	fmt.Printf("Linear regression result: y = %.6fx + %.6f\n", m, b)
	fmt.Println("Expected: y = 2.000000x + 1.000000")
}
