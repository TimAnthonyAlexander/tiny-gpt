package main

import (
	"bufio"
	"crypto/sha256"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Configuration structures
type TrainConfig struct {
	Corpus  string
	Out     string
	Win     int
	Dim     int
	Hidden  int
	Batch   int
	Dropout float64
	L2      float64
	Epochs  int
	LR      float64
	Clip    float64
	Vocab   int
	Seed    int64
}

type InferConfig struct {
	Model string
	Vocab string
	Win   int
	Temp  float64
	TopK  int
	TopP  float64
	Rep   float64
	Max   int
	Seed  int64
}

// Metadata structures
type Manifest struct {
	CorpusPath   string    `json:"corpus_path"`
	CorpusHash   string    `json:"corpus_hash"`
	Win          int       `json:"win"`
	Dim          int       `json:"dim"`
	Hidden       int       `json:"hidden"`
	Batch        int       `json:"batch"`
	Dropout      float64   `json:"dropout"`
	L2           float64   `json:"l2"`
	Epochs       int       `json:"epochs"`
	LR           float64   `json:"lr"`
	Clip         float64   `json:"clip"`
	VocabSize    int       `json:"vocab_size"`
	Seed         int64     `json:"seed"`
	TrainedAt    time.Time `json:"trained_at"`
	BuildVersion string    `json:"build_version"`
}

type Metrics struct {
	Epochs []EpochMetrics `json:"epochs"`
}

type EpochMetrics struct {
	Epoch      int     `json:"epoch"`
	TrainLoss  float64 `json:"train_loss"`
	ValLoss    float64 `json:"val_loss"`
	Perplexity float64 `json:"perplexity"`
}

type VocabData struct {
	ToID   map[string]int `json:"to_id"`
	ToWord map[int]string `json:"to_word"`
	Size   int            `json:"size"`
}

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	command := os.Args[1]
	switch command {
	case "train":
		runTrain(os.Args[2:])
	case "infer":
		runInfer(os.Args[2:])
	case "demo":
		// Keep the old demo functionality
		runDemo()
	default:
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Println("TinyGPT - Minimal Character-Level Language Model")
	fmt.Println()
	fmt.Println("Usage:")
	fmt.Println("  tinygpt train --corpus FILE --out DIR [options]")
	fmt.Println("  tinygpt infer --model FILE --vocab FILE [options]")
	fmt.Println("  tinygpt demo")
	fmt.Println()
	fmt.Println("Commands:")
	fmt.Println("  train    Train a new model from text corpus")
	fmt.Println("  infer    Generate text from stdin using trained model")
	fmt.Println("  demo     Run interactive demo with built-in examples")
}

func runTrain(args []string) {
	fs := flag.NewFlagSet("train", flag.ExitOnError)

	config := TrainConfig{}
	fs.StringVar(&config.Corpus, "corpus", "", "Path to training corpus (required)")
	fs.StringVar(&config.Out, "out", "", "Output directory for model (required)")
	fs.IntVar(&config.Win, "win", 16, "Context window size")
	fs.IntVar(&config.Dim, "dim", 16, "Embedding dimension")
	fs.IntVar(&config.Hidden, "hidden", 64, "Hidden layer size")
	fs.IntVar(&config.Batch, "batch", 256, "Batch size")
	fs.Float64Var(&config.Dropout, "dropout", 0.2, "Dropout probability")
	fs.Float64Var(&config.L2, "l2", 1e-5, "L2 regularization")
	fs.IntVar(&config.Epochs, "epochs", 30, "Number of epochs")
	fs.Float64Var(&config.LR, "lr", 1e-3, "Learning rate")
	fs.Float64Var(&config.Clip, "clip", 1.0, "Gradient clipping")
	fs.IntVar(&config.Vocab, "vocab", 128, "Vocabulary size")
	fs.Int64Var(&config.Seed, "seed", 1337, "Random seed")

	fs.Parse(args)

	if config.Corpus == "" || config.Out == "" {
		fmt.Println("Error: --corpus and --out are required")
		fs.PrintDefaults()
		os.Exit(1)
	}

	trainModel(config)
}

func runInfer(args []string) {
	fs := flag.NewFlagSet("infer", flag.ExitOnError)

	config := InferConfig{}
	fs.StringVar(&config.Model, "model", "", "Path to model file (required)")
	fs.StringVar(&config.Vocab, "vocab", "", "Path to vocab file (required)")
	fs.IntVar(&config.Win, "win", 16, "Context window size")
	fs.Float64Var(&config.Temp, "temp", 0.8, "Temperature")
	fs.IntVar(&config.TopK, "topk", 40, "Top-k sampling")
	fs.Float64Var(&config.TopP, "topp", 0.0, "Top-p (nucleus) sampling")
	fs.Float64Var(&config.Rep, "rep", 0.2, "Repetition penalty")
	fs.IntVar(&config.Max, "max", 200, "Maximum tokens to generate")
	fs.Int64Var(&config.Seed, "seed", 0, "Random seed (0 for random)")

	fs.Parse(args)

	if config.Model == "" || config.Vocab == "" {
		fmt.Println("Error: --model and --vocab are required")
		fs.PrintDefaults()
		os.Exit(1)
	}

	inferFromStdin(config)
}

func runDemo() {
	// Run the existing demo
	RunContextTinyGPT()
}

// Training implementation
func trainModel(config TrainConfig) {
	fmt.Printf("🤖 TinyGPT Training\n")
	fmt.Printf("==================\n\n")

	// Set seed for reproducibility
	rand.Seed(config.Seed)

	// Create output directory
	if err := os.MkdirAll(config.Out, 0755); err != nil {
		fmt.Printf("Error creating output directory: %v\n", err)
		os.Exit(1)
	}

	// Load and preprocess corpus
	fmt.Printf("📚 Loading corpus from %s...\n", config.Corpus)
	corpus, err := loadAndPreprocessCorpus(config.Corpus)
	if err != nil {
		fmt.Printf("Error loading corpus: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("   Corpus length: %d characters\n", len(corpus))

	// Compute corpus hash
	corpusHash := fmt.Sprintf("%x", sha256.Sum256([]byte(corpus)))[:16]
	fmt.Printf("   Corpus hash: %s\n", corpusHash)

	// Build tokenizer
	fmt.Printf("\n📝 Building vocabulary (max %d tokens)...\n", config.Vocab)
	tokenizer := NewTokenizer()
	tokenizer.BuildVocab(corpus, config.Vocab)

	vocabSize := tokenizer.GetVocabSize()
	padID := tokenizer.toID["<pad>"]

	fmt.Printf("   Actual vocabulary size: %d\n", vocabSize)
	fmt.Printf("   Pad ID: %d\n", padID)

	// Check for high UNK rate
	tokens := tokenizer.Encode(corpus)
	unkID := tokenizer.toID["<unk>"]
	unkCount := 0
	for _, token := range tokens {
		if token == unkID {
			unkCount++
		}
	}
	unkRate := float64(unkCount) / float64(len(tokens))
	fmt.Printf("   UNK rate: %.2f%%\n", unkRate*100)

	if unkRate > 0.1 {
		fmt.Printf("   ⚠️  High UNK rate! Consider increasing --vocab\n")
	}

	// Convert to sequences
	fmt.Printf("\n🔢 Creating training sequences...\n")
	sentences := strings.Split(corpus, "\n")
	var sequences [][]int

	for _, sentence := range sentences {
		sentence = strings.TrimSpace(sentence)
		if len(sentence) > 0 {
			tokens := tokenizer.Encode(sentence)
			if len(tokens) > 2 { // Must have more than just start/end
				sequences = append(sequences, tokens)
			}
		}
	}

	fmt.Printf("   Training sequences: %d\n", len(sequences))

	if len(sequences) == 0 {
		fmt.Printf("Error: No training sequences found\n")
		os.Exit(1)
	}

	// Configure training
	trainCfg := TrainCfg{
		Dim:      config.Dim,
		Win:      config.Win,
		Hid:      config.Hidden,
		Batch:    config.Batch,
		DropProb: float32(config.Dropout),
		L2:       float32(config.L2),
		Epochs:   config.Epochs,
		Adam: AdamCfg{
			LR:    float32(config.LR),
			Beta1: 0.9,
			Beta2: 0.999,
			Eps:   1e-8,
			Clip:  float32(config.Clip),
		},
		PadID: padID,
	}

	// Calculate parameter count
	totalParams := vocabSize*config.Dim + config.Dim*config.Win*config.Hidden + config.Hidden + config.Hidden*vocabSize + vocabSize
	fmt.Printf("\n🧠 Model configuration:\n")
	fmt.Printf("   Architecture: Context MLP\n")
	fmt.Printf("   Parameters: %d\n", totalParams)
	fmt.Printf("   Context window: %d tokens\n", config.Win)
	fmt.Printf("   Embedding dim: %d\n", config.Dim)
	fmt.Printf("   Hidden dim: %d\n", config.Hidden)

	// Train the model
	fmt.Printf("\n🏋️  Training for %d epochs...\n", config.Epochs)
	model, finalLoss := TrainWithMetrics(sequences, vocabSize, padID, trainCfg, config.Out)

	fmt.Printf("\n✅ Training complete!\n")
	fmt.Printf("   Final validation loss: %.4f\n", finalLoss)
	fmt.Printf("   Final perplexity: %.2f\n", math.Exp(finalLoss))

	// Save model
	modelPath := filepath.Join(config.Out, "model.gob")
	if err := SaveParams(modelPath, model); err != nil {
		fmt.Printf("Error saving model: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("💾 Model saved to: %s\n", modelPath)

	// Save vocabulary
	vocabPath := filepath.Join(config.Out, "vocab.json")
	vocabData := VocabData{
		ToID:   tokenizer.toID,
		ToWord: tokenizer.toWord,
		Size:   vocabSize,
	}
	if err := saveJSON(vocabPath, vocabData); err != nil {
		fmt.Printf("Error saving vocabulary: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("📝 Vocabulary saved to: %s\n", vocabPath)

	// Save manifest
	manifestPath := filepath.Join(config.Out, "manifest.json")
	manifest := Manifest{
		CorpusPath:   config.Corpus,
		CorpusHash:   corpusHash,
		Win:          config.Win,
		Dim:          config.Dim,
		Hidden:       config.Hidden,
		Batch:        config.Batch,
		Dropout:      config.Dropout,
		L2:           config.L2,
		Epochs:       config.Epochs,
		LR:           config.LR,
		Clip:         config.Clip,
		VocabSize:    vocabSize,
		Seed:         config.Seed,
		TrainedAt:    time.Now(),
		BuildVersion: "dev",
	}
	if err := saveJSON(manifestPath, manifest); err != nil {
		fmt.Printf("Error saving manifest: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("📋 Manifest saved to: %s\n", manifestPath)

	// Quick quality test
	fmt.Printf("\n🎲 Quick quality test:\n")
	testGeneration(model, tokenizer, config.Win, padID)

	fmt.Printf("\n🚀 Training complete! Use the model with:\n")
	fmt.Printf("   echo \"your prompt\" | ./tinygpt infer --model %s --vocab %s\n",
		modelPath, vocabPath)
}

// Inference implementation
func inferFromStdin(config InferConfig) {
	// Set seed if specified
	if config.Seed != 0 {
		rand.Seed(config.Seed)
	}

	// Load model
	model, err := LoadParams(config.Model)
	if err != nil {
		fmt.Printf("Error loading model: %v\n", err)
		os.Exit(1)
	}

	// Load vocabulary
	var vocabData VocabData
	if err := loadJSON(config.Vocab, &vocabData); err != nil {
		fmt.Printf("Error loading vocabulary: %v\n", err)
		os.Exit(1)
	}

	// Reconstruct tokenizer
	tokenizer := &Tokenizer{
		toID:      vocabData.ToID,
		toWord:    vocabData.ToWord,
		vocabSize: vocabData.Size,
	}

	padID := tokenizer.toID["<pad>"]

	// Generation config
	genCfg := GenCfg{
		Win:               config.Win,
		Temp:              float32(config.Temp),
		TopK:              config.TopK,
		TopP:              float32(config.TopP),
		RepetitionPenalty: float32(config.Rep),
		PadID:             padID,
	}

	// Read from stdin line by line
	scanner := bufio.NewScanner(os.Stdin)
	for scanner.Scan() {
		prompt := strings.TrimSpace(scanner.Text())
		if len(prompt) == 0 {
			continue
		}

		// Encode prompt (handle truncation if too long)
		prefixTokens := tokenizer.Encode(prompt)
		if len(prefixTokens) > config.Win {
			prefixTokens = prefixTokens[len(prefixTokens)-config.Win:]
		}

		// Generate completion
		generated := Generate(model, prefixTokens, config.Max, genCfg)
		completion := tokenizer.Decode(generated)

		// Print just the completion (remove the original prompt)
		originalLength := len(tokenizer.Decode(prefixTokens))
		if originalLength < len(completion) {
			result := completion[originalLength:]
			// Stop at <end> token if present
			if endPos := strings.Index(result, "<end>"); endPos >= 0 {
				result = result[:endPos]
			}
			fmt.Println(strings.TrimSpace(result))
		} else {
			fmt.Println() // Empty line if no generation
		}
	}

	if err := scanner.Err(); err != nil {
		fmt.Printf("Error reading stdin: %v\n", err)
		os.Exit(1)
	}
}

// Helper functions
func loadAndPreprocessCorpus(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}

	text := string(data)

	// Normalize text
	text = strings.ToLower(text)

	// Collapse whitespace but preserve structure
	lines := strings.Split(text, "\n")
	var cleanLines []string

	for _, line := range lines {
		// Collapse internal whitespace
		line = strings.Join(strings.Fields(line), " ")
		cleanLines = append(cleanLines, line)
	}

	return strings.Join(cleanLines, "\n"), nil
}

func testGeneration(model *Params, tokenizer *Tokenizer, win int, padID int) {
	gcfg := GenCfg{
		Win:               win,
		Temp:              0.8,
		TopK:              20,
		TopP:              0.0,
		RepetitionPenalty: 0.3,
		PadID:             padID,
	}

	testPrefixes := []string{"the", "and", "in", "to"}

	for _, prefix := range testPrefixes {
		prefixTokens := tokenizer.Encode(prefix)
		generated := Generate(model, prefixTokens, 30, gcfg)
		text := tokenizer.Decode(generated)
		fmt.Printf("   \"%s\" → \"%s\"\n", prefix, text)
	}
}

func saveJSON(path string, data interface{}) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	encoder := json.NewEncoder(f)
	encoder.SetIndent("", "  ")
	return encoder.Encode(data)
}

func loadJSON(path string, data interface{}) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	return json.NewDecoder(f).Decode(data)
}
