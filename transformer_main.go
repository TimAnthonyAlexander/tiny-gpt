package main

import (
	"fmt"
	"strings"
)

func RunTransformerTinyGPT() {
	fmt.Println("🤖 Transformer TinyGPT - Single-Head Causal Self-Attention")
	fmt.Println("==========================================================")

	// Extended training corpus with more patterns and structure
	corpus := `the quick brown fox jumps over the lazy dog. the cat sat on the mat and purred contentedly. hello world! how are you today my friend? the sun is shining bright across the blue sky. birds are singing in the tall green trees. life is beautiful and full of wonder and joy. the ocean waves crash against the rocky shore with great force. mountains stand tall and proud in the distance. rivers flow gently through the peaceful valleys below. flowers bloom in spring with vibrant colors. winter brings snow and ice to the land. summer is warm and sunny and perfect for outdoor activities. autumn leaves fall gently to the ground in shades of red and gold. time moves forward always without stopping. love conquers all fears and doubts. hope lights the way through darkness. dreams come true sometimes if you believe. hard work pays off in the end. knowledge is power indeed and wisdom is precious. the story begins once upon a time in a land far away. characters develop through trials and tribulations. plots thicken with mystery and intrigue. endings bring resolution and closure. words have meaning and power. sentences form thoughts and ideas. paragraphs build arguments and narratives. language is the tool of communication and expression. creativity flows from imagination and inspiration.`

	fmt.Printf("Training corpus length: %d characters\n", len(corpus))

	// Initialize tokenizer and build vocabulary
	fmt.Println("\n📝 Building vocabulary...")
	tokenizer := NewTokenizer()
	tokenizer.BuildVocab(corpus, 128) // Larger vocab for transformer

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

	// Configure the transformer model
	fmt.Println("\n🧠 Configuring transformer model...")
	cfg := TransformerCfg{
		Dim:      16,   // Embedding dimension (d)
		Win:      16,   // Context window
		Hidden:   32,   // MLP hidden size
		Batch:    32,   // Smaller batch for transformer
		DropProb: 0.1,  // Dropout probability
		L2:       1e-5, // Weight decay
		Epochs:   30,   // Number of epochs
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
	d, h := cfg.Dim, cfg.Hidden
	embedParams := vocabSize * d
	attnParams := 4 * d * d // Wq, Wk, Wv, Wo
	mlpParams := d*h + h + h*d + d
	lnParams := 4 * d         // 2 LayerNorms × 2 params each
	outputParams := vocabSize // output bias (embeddings are tied)

	totalParams := embedParams + attnParams + mlpParams + lnParams + outputParams

	fmt.Printf("Transformer architecture:\n")
	fmt.Printf("  Embedding: %d x %d = %d params\n", vocabSize, d, embedParams)
	fmt.Printf("  Attention: 4 x (%d x %d) = %d params\n", d, d, attnParams)
	fmt.Printf("  MLP: %d params\n", mlpParams)
	fmt.Printf("  LayerNorm: %d params\n", lnParams)
	fmt.Printf("  Output bias: %d params\n", outputParams)
	fmt.Printf("  Total parameters: %d\n", totalParams)
	fmt.Printf("  Context window: %d tokens\n", cfg.Win)
	fmt.Printf("  Single attention head: %d dimensions\n", d)

	if totalParams > 20000 {
		fmt.Printf("  ⚠️  Warning: Parameter count exceeds 20k target!\n")
	} else {
		fmt.Printf("  ✅ Parameter count within 20k target!\n")
	}

	// For now, create a simplified demo since full transformer training needs more implementation
	fmt.Println("\n🏗️ Creating transformer model...")
	model := NewTransformerParams(vocabSize, cfg.Dim, cfg.Hidden)

	fmt.Println("\n🎯 Testing transformer architecture...")

	// Test a simple forward pass
	testSample := Sample{
		Context: make([]int, cfg.Win),
		Target:  startID,
	}

	// Fill context with padding
	for i := range testSample.Context {
		testSample.Context[i] = padID
	}

	testBatch := Batch{
		Contexts: [][]int{testSample.Context},
		Targets:  []int{testSample.Target},
		Mask:     []float32{1.0},
	}

	// Test forward pass
	cache, logits := TransformerForward(model, testBatch, cfg.Dim, cfg.Win, false, 0)
	_ = cache

	fmt.Printf("✅ Forward pass successful! Logits shape: %d\n", len(logits[0]))

	// Test generation
	fmt.Println("\n🎲 Testing generation (untrained model)...")

	gcfg := GenCfg{
		Win:               cfg.Win,
		Temp:              1.2,
		TopK:              20,
		TopP:              0.0,
		RepetitionPenalty: 0.1,
		PadID:             padID,
	}

	testPrefixes := []string{"the", "hello"}

	for _, prefix := range testPrefixes {
		prefixTokens := tokenizer.Encode(prefix)
		generated := TransformerGenerate(model, prefixTokens, 40, gcfg)
		text := tokenizer.Decode(generated)
		fmt.Printf("Prefix \"%s\" → %q\n", prefix, text)
	}

	// Save the untrained model for testing
	fmt.Println("\n💾 Saving untrained transformer model...")
	err := SaveTransformerParams("tinygpt_transformer_untrained.gob", model)
	if err != nil {
		fmt.Printf("Error saving model: %v\n", err)
	} else {
		fmt.Println("Untrained model saved to tinygpt_transformer_untrained.gob")
	}

	fmt.Println("\n✅ Transformer TinyGPT architecture test complete!")
	fmt.Println("\n📊 Summary:")
	fmt.Printf("- Architecture: Single-head causal self-attention transformer\n")
	fmt.Printf("- Vocabulary: %d characters\n", vocabSize)
	fmt.Printf("- Embedding dimension: %d\n", d)
	fmt.Printf("- Context window: %d tokens\n", cfg.Win)
	fmt.Printf("- MLP hidden size: %d\n", h)
	fmt.Printf("- Parameters: %d (target: <20k)\n", totalParams)
	fmt.Printf("- Features: Positional encoding, LayerNorm, GELU, Weight tying\n")
	fmt.Printf("- Components: Embed → LN → Attn → Residual → LN → MLP → Residual → Output\n")

	fmt.Println("\n📝 Next steps:")
	fmt.Println("- Implement full backward pass for all transformer components")
	fmt.Println("- Add proper batch processing")
	fmt.Println("- Train the model with teacher forcing")
	fmt.Println("- Test with larger context windows and vocabulary")
	fmt.Println("- Add multiple transformer blocks")

	fmt.Println("\n🚀 This is a major upgrade from context MLP to full transformer!")
}
