package main

func main() {
	// Run the context-aware version with proper autoregressive training
	RunContextTinyGPT()
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
