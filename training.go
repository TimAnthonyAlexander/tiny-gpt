package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
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

// Softmax computes softmax probabilities from logits
func Softmax(logits []float32) []float32 {
	max := logits[0]
	for _, v := range logits[1:] {
		if v > max {
			max = v
		}
	}
	
	var sum float32
	result := make([]float32, len(logits))
	
	for i, v := range logits {
		exp := float32(math.Exp(float64(v - max)))
		result[i] = exp
		sum += exp
	}
	
	for i := range result {
		result[i] /= sum
	}
	
	return result
}

// WeightedChoice samples from probability distribution
func WeightedChoice(probs []float32) int {
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

// Train trains the model on the given dataset
func Train(model *TinyLM, trainData []TrainingData, epochs int, learningRate float64) error {
	g := model.g
	
	// Create input/output placeholders
	x := gorgonia.NewScalar(g, tensor.Int, gorgonia.WithName("input"))
	y := gorgonia.NewScalar(g, tensor.Int, gorgonia.WithName("target"))
	
	// Forward pass
	logits, err := model.Forward(x)
	if err != nil {
		return fmt.Errorf("forward pass failed: %w", err)
	}
	
	// Convert target to one-hot for cross-entropy
	vocabSize := model.w2.Shape()[1]
	targetOneHot := gorgonia.NewMatrix(g, tensor.Float32, gorgonia.WithShape(1, vocabSize))
	
	// For simplicity, we'll use a basic loss computation
	// In practice, you'd want proper sparse categorical cross-entropy
	probs, err := gorgonia.SoftMax(logits)
	if err != nil {
		return fmt.Errorf("softmax failed: %w", err)
	}
	
	// Simple loss - we'll implement a basic version
	// Create target one-hot manually in the training loop
	
	// Create tape machine and solver
	vm := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(model.GetLearnables()...))
	solver := gorgonia.NewAdamSolver(gorgonia.WithLearnRate(learningRate))
	
	defer vm.Close()
	
	fmt.Printf("Starting training for %d epochs...\n", epochs)
	
	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0
		
		// Shuffle training data
		shuffled := make([]TrainingData, len(trainData))
		copy(shuffled, trainData)
		rand.Shuffle(len(shuffled), func(i, j int) {
			shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
		})
		
		for i, sample := range shuffled {
			// Set input and target
			err := gorgonia.Let(x, sample.Input)
			if err != nil {
				return fmt.Errorf("setting input failed: %w", err)
			}
			
			// Create target one-hot vector
			targetVec := tensor.New(tensor.WithShape(vocabSize), tensor.WithBacking(make([]float32, vocabSize)))
			targetData := targetVec.Data().([]float32)
			targetData[sample.Target] = 1.0
			
			err = gorgonia.Let(targetOneHot, targetVec)
			if err != nil {
				return fmt.Errorf("setting target failed: %w", err)
			}
			
			// Reset and run
			vm.Reset()
			err = vm.RunAll()
			if err != nil {
				return fmt.Errorf("vm.RunAll failed at epoch %d, sample %d: %w", epoch, i, err)
			}
			
			// Compute loss manually (cross-entropy)
			probsVal := probs.Value()
			if probsVal == nil {
				return fmt.Errorf("probs value is nil")
			}
			
			probsData := probsVal.Data().([]float32)
			loss := -math.Log(float64(probsData[sample.Target]) + 1e-8)
			totalLoss += loss
			
			// Update parameters
			err = solver.Step(gorgonia.NodesToValueGrads(model.GetLearnables()))
			if err != nil {
				return fmt.Errorf("solver step failed: %w", err)
			}
		}
		
		avgLoss := totalLoss / float64(len(shuffled))
		if epoch%10 == 0 || epoch == epochs-1 {
			fmt.Printf("Epoch %d: avg loss = %.4f\n", epoch, avgLoss)
		}
	}
	
	fmt.Println("Training completed!")
	return nil
}

// Sample generates text by sampling from the model
func Sample(model *TinyLM, tokenizer *Tokenizer, startToken int, maxLength int, temperature float32) ([]int, error) {
	g := gorgonia.NewGraph()
	
	// Rebuild model for inference (without gradients)
	inferModel := NewTinyLM(g, tokenizer.GetVocabSize(), 16, 32)
	
	// Copy weights from trained model (this is simplified - in practice you'd save/load weights)
	// For now, we'll assume the model is already trained and ready to use
	
	x := gorgonia.NewScalar(g, tensor.Int, gorgonia.WithName("input"))
	logits, err := inferModel.Forward(x)
	if err != nil {
		return nil, fmt.Errorf("forward pass failed: %w", err)
	}
	
	vm := gorgonia.NewTapeMachine(g)
	defer vm.Close()
	
	sequence := []int{startToken}
	currentToken := startToken
	
	for len(sequence) < maxLength {
		err := gorgonia.Let(x, currentToken)
		if err != nil {
			return nil, fmt.Errorf("setting input failed: %w", err)
		}
		
		vm.Reset()
		err = vm.RunAll()
		if err != nil {
			return nil, fmt.Errorf("vm.RunAll failed: %w", err)
		}
		
		logitsVal := logits.Value()
		if logitsVal == nil {
			return nil, fmt.Errorf("logits value is nil")
		}
		
		logitsData := logitsVal.Data().([]float32)
		
		// Apply temperature
		for i := range logitsData {
			logitsData[i] /= temperature
		}
		
		probs := Softmax(logitsData)
		nextToken := WeightedChoice(probs)
		
		// Stop if we hit end token
		if endID, exists := tokenizer.toID["<end>"]; exists && nextToken == endID {
			break
		}
		
		sequence = append(sequence, nextToken)
		currentToken = nextToken
	}
	
	return sequence, nil
}

func init() {
	rand.Seed(time.Now().UnixNano())
}
