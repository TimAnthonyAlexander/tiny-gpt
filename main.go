package main

import (
	"fmt"
	"log"

	"gorgonia.org/gorgonia"
)

func main() {
	fmt.Println("🤖 TinyGPT - Minimal GPT-style Autocompleter")
	fmt.Println("============================================")
	
	// Sample training corpus - simple patterns for the model to learn
	corpus := `the quick brown fox jumps over the lazy dog. the cat sat on the mat. hello world! how are you today? the sun is shining bright. birds are singing in the trees. life is beautiful and full of wonder. the ocean waves crash against the shore. mountains stand tall and proud. rivers flow gently through the valleys. flowers bloom in spring. winter brings snow and ice. summer is warm and sunny. autumn leaves fall gently. time moves forward always. love conquers all fears. hope lights the way. dreams come true sometimes. hard work pays off. knowledge is power indeed.`
	
	fmt.Printf("Training corpus length: %d characters\n", len(corpus))
	
	// Initialize tokenizer and build vocabulary
	fmt.Println("\n📝 Building vocabulary...")
	tokenizer := NewTokenizer()
	tokenizer.BuildVocab(corpus, 200) // Small vocab for this toy example
	
	vocabSize := tokenizer.GetVocabSize()
	fmt.Printf("Vocabulary size: %d\n", vocabSize)
	
	// Show some vocab examples
	fmt.Println("Sample vocabulary:")
	for i := 0; i < 10 && i < vocabSize; i++ {
		if word, exists := tokenizer.toWord[i]; exists {
			fmt.Printf("  %d -> %q\n", i, word)
		}
	}
	
	// Tokenize the corpus
	fmt.Println("\n🔢 Tokenizing corpus...")
	tokens := tokenizer.Encode(corpus)
	fmt.Printf("Token sequence length: %d\n", len(tokens))
	fmt.Printf("First 20 tokens: %v\n", tokens[:min(20, len(tokens))])
	
	// Create training data
	fmt.Println("\n📊 Creating training data...")
	trainData := CreateTrainingData(tokens)
	fmt.Printf("Training samples: %d\n", len(trainData))
	
	// Show some training examples
	fmt.Println("Sample training pairs:")
	for i := 0; i < min(5, len(trainData)); i++ {
		inputChar := tokenizer.toWord[trainData[i].Input]
		targetChar := tokenizer.toWord[trainData[i].Target]
		fmt.Printf("  %q -> %q\n", inputChar, targetChar)
	}
	
	// Initialize model
	fmt.Println("\n🧠 Initializing model...")
	g := gorgonia.NewGraph()
	
	embeddingDim := 16
	hiddenDim := 32
	model := NewTinyLM(g, vocabSize, embeddingDim, hiddenDim)
	
	// Calculate parameter count
	totalParams := vocabSize*embeddingDim + embeddingDim*hiddenDim + hiddenDim + hiddenDim*vocabSize + vocabSize
	fmt.Printf("Model parameters: %d\n", totalParams)
	fmt.Printf("Architecture: %d -> %d -> %d -> %d\n", vocabSize, embeddingDim, hiddenDim, vocabSize)
	
	// Train the model
	fmt.Println("\n🏋️ Training model...")
	epochs := 100
	learningRate := 0.01
	
	err := Train(model, trainData, epochs, learningRate)
	if err != nil {
		log.Fatalf("Training failed: %v", err)
	}
	
	// Generate some text samples
	fmt.Println("\n🎯 Generating samples...")
	
	startTokenID := tokenizer.toID["<start>"]
	
	for i := 0; i < 3; i++ {
		fmt.Printf("\nSample %d:\n", i+1)
		
		sequence, err := Sample(model, tokenizer, startTokenID, 50, 1.0)
		if err != nil {
			fmt.Printf("Sampling failed: %v\n", err)
			continue
		}
		
		generated := tokenizer.Decode(sequence)
		fmt.Printf("Generated: %q\n", generated)
	}
	
	// Interactive mode simulation
	fmt.Println("\n💬 Autocomplete examples:")
	testPrefixes := []string{"the", "h", "cat", "sun"}
	
	for _, prefix := range testPrefixes {
		fmt.Printf("\nPrefix: %q\n", prefix)
		prefixTokens := tokenizer.Encode(prefix)
		if len(prefixTokens) > 0 {
			// Use last token of prefix as starting point
			lastToken := prefixTokens[len(prefixTokens)-1]
			sequence, err := Sample(model, tokenizer, lastToken, 20, 0.8)
			if err != nil {
				fmt.Printf("Error: %v\n", err)
				continue
			}
			completion := tokenizer.Decode(sequence)
			fmt.Printf("Autocomplete: %q\n", completion)
		}
	}
	
	fmt.Println("\n✅ TinyGPT demonstration complete!")
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
