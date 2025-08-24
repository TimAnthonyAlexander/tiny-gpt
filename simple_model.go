package main

import (
	"fmt"
	"math"
	"math/rand"
)

type TrainingData struct {
	Input  int
	Target int
}

// CreateTrainingData generates training pairs from tokenized sequence
func CreateTrainingData(tokens []int) []TrainingData {
	var data []TrainingData
	
	// Create pairs: predict next token given current token
	for i := 0; i < len(tokens)-1; i++ {
		data = append(data, TrainingData{
			Input:  tokens[i],
			Target: tokens[i+1],
		})
	}
	
	return data
}

// SimpleTinyGPT - A simplified version without Gorgonia for learning purposes
type SimpleTinyGPT struct {
	// Embedding matrix: vocab_size x embedding_dim
	embeddings [][]float32
	
	// First layer: embedding_dim x hidden_dim  
	w1 [][]float32
	b1 []float32
	
	// Output layer: hidden_dim x vocab_size
	w2 [][]float32
	b2 []float32
	
	vocabSize    int
	embeddingDim int
	hiddenDim    int
}

func NewSimpleTinyGPT(vocabSize, embeddingDim, hiddenDim int) *SimpleTinyGPT {
	model := &SimpleTinyGPT{
		vocabSize:    vocabSize,
		embeddingDim: embeddingDim,
		hiddenDim:    hiddenDim,
	}
	
	// Initialize embeddings
	model.embeddings = make([][]float32, vocabSize)
	for i := range model.embeddings {
		model.embeddings[i] = make([]float32, embeddingDim)
		for j := range model.embeddings[i] {
			model.embeddings[i][j] = rand.Float32()*0.2 - 0.1 // Xavier init
		}
	}
	
	// Initialize W1 and b1
	model.w1 = make([][]float32, embeddingDim)
	for i := range model.w1 {
		model.w1[i] = make([]float32, hiddenDim)
		for j := range model.w1[i] {
			model.w1[i][j] = rand.Float32()*0.2 - 0.1
		}
	}
	model.b1 = make([]float32, hiddenDim)
	
	// Initialize W2 and b2
	model.w2 = make([][]float32, hiddenDim)
	for i := range model.w2 {
		model.w2[i] = make([]float32, vocabSize)
		for j := range model.w2[i] {
			model.w2[i][j] = rand.Float32()*0.2 - 0.1
		}
	}
	model.b2 = make([]float32, vocabSize)
	
	return model
}

// Forward pass
func (m *SimpleTinyGPT) Forward(tokenID int) []float32 {
	if tokenID >= m.vocabSize {
		tokenID = 0 // fallback
	}
	
	// Embedding lookup
	x := make([]float32, m.embeddingDim)
	copy(x, m.embeddings[tokenID])
	
	// First layer: x * W1 + b1
	h := make([]float32, m.hiddenDim)
	for j := 0; j < m.hiddenDim; j++ {
		sum := m.b1[j]
		for i := 0; i < m.embeddingDim; i++ {
			sum += x[i] * m.w1[i][j]
		}
		// ReLU activation
		if sum > 0 {
			h[j] = sum
		}
	}
	
	// Output layer: h * W2 + b2
	logits := make([]float32, m.vocabSize)
	for j := 0; j < m.vocabSize; j++ {
		sum := m.b2[j]
		for i := 0; i < m.hiddenDim; i++ {
			sum += h[i] * m.w2[i][j]
		}
		logits[j] = sum
	}
	
	return logits
}

// Simple backpropagation and parameter update
func (m *SimpleTinyGPT) Train(trainData []TrainingData, epochs int, learningRate float32) {
	fmt.Printf("Training simple model for %d epochs...\n", epochs)
	
	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0
		
		// Shuffle training data
		shuffled := make([]TrainingData, len(trainData))
		copy(shuffled, trainData)
		rand.Shuffle(len(shuffled), func(i, j int) {
			shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
		})
		
		for _, sample := range shuffled {
			// Forward pass
			logits := m.Forward(sample.Input)
			
			// Softmax
			probs := SimpleSoftmax(logits)
			
			// Cross-entropy loss with better numerical stability
			if sample.Target < len(probs) && probs[sample.Target] > 1e-10 {
				loss := -math.Log(float64(probs[sample.Target]) + 1e-10)
				if !math.IsNaN(loss) && !math.IsInf(loss, 0) {
					totalLoss += loss
				}
			}
			
			// Backward pass (simplified)
			m.simpleBackward(sample.Input, sample.Target, logits, probs, learningRate)
		}
		
		avgLoss := totalLoss / float64(len(shuffled))
		if epoch%20 == 0 || epoch == epochs-1 {
			fmt.Printf("Epoch %d: avg loss = %.4f\n", epoch, avgLoss)
		}
	}
}

func (m *SimpleTinyGPT) simpleBackward(input, target int, logits, probs []float32, lr float32) {
	if input >= m.vocabSize || target >= m.vocabSize {
		return
	}
	
	// Output layer gradients
	outputGrads := make([]float32, m.vocabSize)
	for i := 0; i < m.vocabSize; i++ {
		if i == target {
			outputGrads[i] = probs[i] - 1.0
		} else {
			outputGrads[i] = probs[i]
		}
	}
	
	// Get hidden activations (re-compute for simplicity)
	x := m.embeddings[input]
	h := make([]float32, m.hiddenDim)
	for j := 0; j < m.hiddenDim; j++ {
		sum := m.b1[j]
		for i := 0; i < m.embeddingDim; i++ {
			sum += x[i] * m.w1[i][j]
		}
		if sum > 0 {
			h[j] = sum
		}
	}
	
	// Update output layer with gradient clipping
	for i := 0; i < m.hiddenDim; i++ {
		for j := 0; j < m.vocabSize; j++ {
			grad := outputGrads[j] * h[i]
			// Clip gradients to prevent explosion
			if grad > 5.0 {
				grad = 5.0
			} else if grad < -5.0 {
				grad = -5.0
			}
			m.w2[i][j] -= lr * grad
		}
	}
	for j := 0; j < m.vocabSize; j++ {
		grad := outputGrads[j]
		if grad > 5.0 {
			grad = 5.0
		} else if grad < -5.0 {
			grad = -5.0
		}
		m.b2[j] -= lr * grad
	}
	
	// Hidden layer gradients
	hiddenGrads := make([]float32, m.hiddenDim)
	for i := 0; i < m.hiddenDim; i++ {
		for j := 0; j < m.vocabSize; j++ {
			hiddenGrads[i] += outputGrads[j] * m.w2[i][j]
		}
		// ReLU derivative
		if h[i] <= 0 {
			hiddenGrads[i] = 0
		}
	}
	
	// Update first layer with gradient clipping
	for i := 0; i < m.embeddingDim; i++ {
		for j := 0; j < m.hiddenDim; j++ {
			grad := hiddenGrads[j] * x[i]
			// Clip gradients to prevent explosion
			if grad > 5.0 {
				grad = 5.0
			} else if grad < -5.0 {
				grad = -5.0
			}
			m.w1[i][j] -= lr * grad
		}
	}
	for j := 0; j < m.hiddenDim; j++ {
		grad := hiddenGrads[j]
		if grad > 5.0 {
			grad = 5.0
		} else if grad < -5.0 {
			grad = -5.0
		}
		m.b1[j] -= lr * grad
	}
	
	// Update embeddings with gradient clipping
	embGrads := make([]float32, m.embeddingDim)
	for i := 0; i < m.embeddingDim; i++ {
		for j := 0; j < m.hiddenDim; j++ {
			embGrads[i] += hiddenGrads[j] * m.w1[i][j]
		}
		// Clip embedding gradients
		grad := embGrads[i]
		if grad > 5.0 {
			grad = 5.0
		} else if grad < -5.0 {
			grad = -5.0
		}
		m.embeddings[input][i] -= lr * grad
	}
}

// Generate text by sampling from the model
func (m *SimpleTinyGPT) Generate(startToken int, maxLength int, temperature float32) []int {
	sequence := []int{startToken}
	currentToken := startToken
	
	for len(sequence) < maxLength {
		logits := m.Forward(currentToken)
		
		// Apply temperature
		for i := range logits {
			logits[i] /= temperature
		}
		
		probs := SimpleSoftmax(logits)
		nextToken := SimpleWeightedChoice(probs)
		
		sequence = append(sequence, nextToken)
		currentToken = nextToken
	}
	
	return sequence
}

func SimpleSoftmax(logits []float32) []float32 {
	// Find max for numerical stability
	max := logits[0]
	for _, v := range logits[1:] {
		if v > max {
			max = v
		}
	}
	
	// Compute exp and sum with better numerical stability
	result := make([]float32, len(logits))
	var sum float32
	for i, v := range logits {
		// Clamp the exponent to prevent overflow
		exp_val := float64(v - max)
		if exp_val > 700 { // prevent overflow
			exp_val = 700
		} else if exp_val < -700 { // prevent underflow
			exp_val = -700
		}
		exp := float32(math.Exp(exp_val))
		result[i] = exp
		sum += exp
	}
	
	// Normalize with numerical stability check
	if sum < 1e-10 {
		// If sum is too small, return uniform distribution
		uniform := float32(1.0) / float32(len(logits))
		for i := range result {
			result[i] = uniform
		}
	} else {
		for i := range result {
			result[i] /= sum
			// Ensure probabilities are not too small
			if result[i] < 1e-10 {
				result[i] = 1e-10
			}
		}
	}
	
	return result
}

func SimpleWeightedChoice(probs []float32) int {
	r := rand.Float32()
	var cumsum float32
	
	for i, p := range probs {
		cumsum += p
		if r <= cumsum {
			return i
		}
	}
	
	return len(probs) - 1
}
