# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a Go project that demonstrates gradient descent algorithms. The main implementation shows two examples:
1. Simple function minimization for f(x) = (x - 2)^2 + 3
2. Linear regression using gradient descent to fit a line to data points

## Common Commands

To run the gradient descent examples:
```bash
go run main.go
```

To build the project:
```bash
go build
```

## Code Architecture

The main.go file contains:
- A Point struct for representing 2D coordinates
- A gradientDescent function that implements gradient descent for function minimization
- A linearRegression function that uses gradient descent for linear regression
- A main function with two examples demonstrating the algorithms

The gradient descent implementations follow the standard iterative update formula:
theta = theta - learning_rate * gradient

For linear regression, the gradients are calculated with respect to the slope (m) and y-intercept (b) parameters using mean squared error loss.