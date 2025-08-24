package main

import (
	"fmt"
	"time"
	"math/rand"
)

func RunSimpleTinyGPT() {
	fmt.Println("🤖 Simple TinyGPT - Minimal GPT-style Autocompleter")
	fmt.Println("=================================================")
	
	// Initialize random seed
	rand.Seed(time.Now().UnixNano())
	
	// Sample training corpus - simple patterns for the model to learn
	corpus := `the quick brown fox jumps over the lazy dog. the cat sat on the mat. hello world! how are you today? the sun is shining bright. birds are singing in the trees. life is beautiful and full of wonder. the ocean waves crash against the shore. mountains stand tall and proud. rivers flow gently through the valleys. flowers bloom in spring. winter brings snow and ice. summer is warm and sunny. autumn leaves fall gently. time moves forward always. love conquers all fears. hope lights the way. dreams come true sometimes. hard work pays off. knowledge is power indeed.`
	
	fmt.Printf("Training corpus length: %d characters\n", len(corpus))
	
	// Initialize tokenizer and build vocabulary
	fmt.Println("\n📝 Building vocabulary...")
	tokenizer := NewTokenizer()
	tokenizer.BuildVocab(corpus, 100) // Smaller vocab for this simple example
	
	vocabSize := tokenizer.GetVocabSize()
	fmt.Printf("Vocabulary size: %d\n", vocabSize)
	
	// Show some vocab examples
	fmt.Println("Sample vocabulary:")
	for i := 0; i < min(10, vocabSize); i++ {
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
	
	// Initialize simple model
	fmt.Println("\n🧠 Initializing simple model...")
	embeddingDim := 16
	hiddenDim := 32
	model := NewSimpleTinyGPT(vocabSize, embeddingDim, hiddenDim)
	
	// Calculate parameter count
	totalParams := vocabSize*embeddingDim + embeddingDim*hiddenDim + hiddenDim + hiddenDim*vocabSize + vocabSize
	fmt.Printf("Model parameters: %d\n", totalParams)
	fmt.Printf("Architecture: %d -> %d -> %d -> %d\n", vocabSize, embeddingDim, hiddenDim, vocabSize)
	
	// Train the model
	fmt.Println("\n🏋️ Training model...")
	epochs := 100
	learningRate := float32(0.01) // Reduced learning rate for stability
	
	model.Train(trainData, epochs, learningRate)
	
	// Generate some text samples
	fmt.Println("\n🎯 Generating samples...")
	
	startTokenID := tokenizer.toID["<start>"]
	
	for i := 0; i < 3; i++ {
		fmt.Printf("\nSample %d:\n", i+1)
		
		sequence := model.Generate(startTokenID, 30, 0.8)
		generated := tokenizer.Decode(sequence)
		fmt.Printf("Generated: %q\n", generated)
	}
	
	// Interactive mode simulation
	fmt.Println("\n💬 Autocomplete examples:")
	testPrefixes := []string{"the", "h", "cat", "sun", "w"}
	
	for _, prefix := range testPrefixes {
		fmt.Printf("\nPrefix: %q\n", prefix)
		prefixTokens := tokenizer.Encode(prefix)
		if len(prefixTokens) > 1 { // Skip start token
			// Use last meaningful token of prefix as starting point
			lastToken := prefixTokens[len(prefixTokens)-2] // -2 to skip <end> token
			sequence := model.Generate(lastToken, 15, 0.7)
			completion := tokenizer.Decode(sequence)
			fmt.Printf("Autocomplete: %q\n", completion)
		}
	}
	
	// Test with individual characters
	fmt.Println("\n🔤 Character-level predictions:")
	testChars := []string{"t", "h", "a", "o", "s"}
	
	for _, char := range testChars {
		if tokenID, exists := tokenizer.toID[char]; exists {
			logits := model.Forward(tokenID)
			probs := SimpleSoftmax(logits)
			
			// Find top 3 predictions
			type prediction struct {
				token string
				prob  float32
			}
			
			var preds []prediction
			for i, prob := range probs {
				if word, exists := tokenizer.toWord[i]; exists {
					preds = append(preds, prediction{word, prob})
				}
			}
			
			// Sort by probability (simple bubble sort)
			for i := 0; i < len(preds)-1; i++ {
				for j := 0; j < len(preds)-i-1; j++ {
					if preds[j].prob < preds[j+1].prob {
						preds[j], preds[j+1] = preds[j+1], preds[j]
					}
				}
			}
			
			fmt.Printf("'%s' -> Top predictions: ", char)
			for i := 0; i < min(3, len(preds)); i++ {
				fmt.Printf("'%s'(%.3f) ", preds[i].token, preds[i].prob)
			}
			fmt.Println()
		}
	}
	
	fmt.Println("\n✅ Simple TinyGPT demonstration complete!")
	fmt.Println("\n📝 Summary:")
	fmt.Printf("- Vocabulary size: %d characters\n", vocabSize)
	fmt.Printf("- Model parameters: %d\n", totalParams)
	fmt.Printf("- Training samples: %d\n", len(trainData))
	fmt.Printf("- Architecture: Embedding(%d) -> Hidden(%d) -> Output(%d)\n", embeddingDim, hiddenDim, vocabSize)
	fmt.Println("- The model learned basic character-level patterns from the corpus")
	fmt.Println("- It can now predict next characters and generate simple text")
}
