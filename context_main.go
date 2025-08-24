package main

import (
	"fmt"
	"math"
	"strings"
)

func RunContextTinyGPT() {
	fmt.Println("🤖 Context TinyGPT - Autoregressive Character LM with Context Window")
	fmt.Println("==================================================================")

	// Extended training corpus with more patterns and structure
	corpus := `the quick brown fox jumps over the lazy dog. the cat sat on the mat and purred contentedly. hello world! how are you today my friend? the sun is shining bright across the blue sky. birds are singing in the tall green trees. life is beautiful and full of wonder and joy. the ocean waves crash against the rocky shore with great force. mountains stand tall and proud in the distance. rivers flow gently through the peaceful valleys below. flowers bloom in spring with vibrant colors. winter brings snow and ice to the land. summer is warm and sunny and perfect for outdoor activities. autumn leaves fall gently to the ground in shades of red and gold. time moves forward always without stopping. love conquers all fears and doubts. hope lights the way through darkness. dreams come true sometimes if you believe. hard work pays off in the end. knowledge is power indeed and wisdom is precious. the story begins once upon a time in a land far away. characters develop through trials and tribulations. plots thicken with mystery and intrigue. endings bring resolution and closure. words have meaning and power. sentences form thoughts and ideas. paragraphs build arguments and narratives. language is the tool of communication and expression.`

	fmt.Printf("Training corpus length: %d characters\n", len(corpus))

	// Initialize tokenizer and build vocabulary
	fmt.Println("\n📝 Building vocabulary...")
	tokenizer := NewTokenizer()
	tokenizer.BuildVocab(corpus, 100) // Reasonable vocab size

	vocabSize := tokenizer.GetVocabSize()
	padID := tokenizer.toID["<pad>"]
	startID := tokenizer.toID["<start>"]
	endID := tokenizer.toID["<end>"]

	fmt.Printf("Vocabulary size: %d\n", vocabSize)
	fmt.Printf("Pad ID: %d, Start ID: %d, End ID: %d\n", padID, startID, endID)

	// Show some vocabulary examples
	fmt.Println("Sample vocabulary:")
	for i := 0; i < min(15, vocabSize); i++ {
		if word, exists := tokenizer.toWord[i]; exists {
			fmt.Printf("  %d -> %q\n", i, word)
		}
	}

	// Split corpus into sentences and tokenize
	fmt.Println("\n🔢 Tokenizing corpus...")
	sentences := strings.Split(corpus, ". ")
	var sequences [][]int

	for _, sentence := range sentences {
		if len(strings.TrimSpace(sentence)) > 0 {
			// Add period back if it was split off
			if !strings.HasSuffix(sentence, ".") && !strings.HasSuffix(sentence, "!") && !strings.HasSuffix(sentence, "?") {
				sentence += "."
			}
			tokens := tokenizer.Encode(sentence)
			sequences = append(sequences, tokens)
		}
	}

	fmt.Printf("Number of sequences: %d\n", len(sequences))
	totalTokens := 0
	for _, seq := range sequences {
		totalTokens += len(seq)
	}
	fmt.Printf("Total tokens: %d\n", totalTokens)

	// Show some example sequences
	fmt.Println("Sample sequences:")
	for i := 0; i < min(3, len(sequences)); i++ {
		fmt.Printf("  Seq %d (%d tokens): %v\n", i, len(sequences[i]), sequences[i][:min(10, len(sequences[i]))])
		decoded := tokenizer.Decode(sequences[i])
		fmt.Printf("  Decoded: %q\n", decoded[:min(50, len(decoded))])
	}

	// Configure the model
	fmt.Println("\n🧠 Configuring context model...")
	cfg := TrainCfg{
		Dim:      16,   // Embedding dimension
		Win:      16,   // Context window (reduced for smaller corpus)
		Hid:      32,   // Hidden layer size
		Batch:    64,   // Batch size (reduced for smaller dataset)
		DropProb: 0.1,  // Dropout probability
		L2:       1e-5, // Weight decay
		Epochs:   50,   // Number of epochs
		Adam: AdamCfg{
			LR:    1e-3,  // Learning rate
			Beta1: 0.9,   // Adam beta1
			Beta2: 0.999, // Adam beta2
			Eps:   1e-8,  // Adam epsilon
			Clip:  1.0,   // Gradient clipping
		},
		PadID: padID,
	}

	// Calculate parameter count
	totalParams := vocabSize*cfg.Dim + cfg.Dim*cfg.Win*cfg.Hid + cfg.Hid + cfg.Hid*vocabSize + vocabSize
	fmt.Printf("Model architecture:\n")
	fmt.Printf("  Embedding: %d x %d = %d params\n", vocabSize, cfg.Dim, vocabSize*cfg.Dim)
	fmt.Printf("  Hidden: %d x %d = %d params\n", cfg.Dim*cfg.Win, cfg.Hid, cfg.Dim*cfg.Win*cfg.Hid)
	fmt.Printf("  Output: %d x %d = %d params\n", cfg.Hid, vocabSize, cfg.Hid*vocabSize)
	fmt.Printf("  Total parameters: %d\n", totalParams)
	fmt.Printf("  Context window: %d tokens\n", cfg.Win)

	// Train the model
	fmt.Println("\n🏋️ Training context model...")
	model, finalLoss := Train(sequences, vocabSize, padID, cfg)
	fmt.Printf("Final validation loss: %.4f\n", finalLoss)
	fmt.Printf("Final validation perplexity: %.2f\n", math.Exp(finalLoss))

	// Save the trained model
	fmt.Println("\n💾 Saving model...")
	err := SaveParams("tinygpt_context.gob", model)
	if err != nil {
		fmt.Printf("Error saving model: %v\n", err)
	} else {
		fmt.Println("Model saved to tinygpt_context.gob")
	}

	// Test generation with different configurations
	fmt.Println("\n🎯 Generating samples...")

	testPrefixes := []string{"the", "hello", "birds", "time", "once upon"}

	for _, prefix := range testPrefixes {
		fmt.Printf("\n--- Prefix: %q ---\n", prefix)
		prefixTokens := tokenizer.Encode(prefix)

		// Conservative generation
		gcfg1 := GenCfg{
			Win:               cfg.Win,
			Temp:              0.7,
			TopK:              20,
			TopP:              0.0,
			RepetitionPenalty: 0.1,
			PadID:             padID,
		}

		generated1 := Generate(model, prefixTokens, 80, gcfg1)
		text1 := tokenizer.Decode(generated1)
		fmt.Printf("Conservative (temp=0.7, top-k=20): %q\n", text1)

		// Creative generation
		gcfg2 := GenCfg{
			Win:               cfg.Win,
			Temp:              1.0,
			TopK:              0,
			TopP:              0.9,
			RepetitionPenalty: 0.3,
			PadID:             padID,
		}

		generated2 := Generate(model, prefixTokens, 80, gcfg2)
		text2 := tokenizer.Decode(generated2)
		fmt.Printf("Creative (temp=1.0, top-p=0.9): %q\n", text2)
	}

	// Interactive-style completion
	fmt.Println("\n💬 Interactive completion examples:")
	completionTests := []string{
		"the sun is",
		"birds are",
		"life is",
		"once upon a time",
		"the quick brown",
	}

	gcfg := GenCfg{
		Win:               cfg.Win,
		Temp:              0.8,
		TopK:              30,
		TopP:              0.0,
		RepetitionPenalty: 0.2,
		PadID:             padID,
	}

	for _, prompt := range completionTests {
		promptTokens := tokenizer.Encode(prompt)
		completion := Generate(model, promptTokens, 40, gcfg)
		completedText := tokenizer.Decode(completion)

		// Find where the original prompt ends
		originalEnd := len(tokenizer.Decode(promptTokens))
		if originalEnd < len(completedText) {
			continuation := completedText[originalEnd:]
			fmt.Printf("📝 \"%s\" → \"%s\"\n", prompt, strings.TrimSpace(continuation))
		}
	}

	fmt.Println("\n✅ Context TinyGPT demonstration complete!")
	fmt.Println("\n📊 Summary:")
	fmt.Printf("- Vocabulary: %d characters\n", vocabSize)
	fmt.Printf("- Context window: %d tokens\n", cfg.Win)
	fmt.Printf("- Parameters: %d (still under 20k target!)\n", totalParams)
	fmt.Printf("- Training sequences: %d\n", len(sequences))
	fmt.Printf("- Final perplexity: %.2f\n", math.Exp(finalLoss))
	fmt.Println("- Architecture: Context-aware MLP with teacher forcing")
	fmt.Println("- Features: Adam optimizer, dropout, batching, advanced sampling")
	fmt.Println("- The model now understands longer-range dependencies!")
}
