package main

func main() {
	// Run the transformer version with causal self-attention
	RunTransformerTinyGPT()
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
