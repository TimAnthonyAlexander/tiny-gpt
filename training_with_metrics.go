package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"time"
)

// TrainWithMetrics trains a model and saves training metrics
func TrainWithMetrics(seqs [][]int, vocab, padID int, cfg TrainCfg, outputDir string) (*Params, float64) {
	rand.Seed(time.Now().UnixNano())

	// Create all window samples from sequences
	var samples []Sample
	for _, s := range seqs {
		samples = append(samples, MakeWindowSamples(s, cfg.Win, padID)...)
	}

	// Split train/validation
	split := int(float64(len(samples)) * 0.9)
	trainS := samples[:split]
	valS := samples[split:]

	fmt.Printf("   Training samples: %d, Validation samples: %d\n", len(trainS), len(valS))

	p := NewParams(vocab, cfg.Dim, cfg.Win, cfg.Hid)
	moms := NewMomentsLike(p)

	// Track metrics
	var allMetrics []EpochMetrics
	bestValLoss := math.Inf(1)
	var bestModel *Params

	for epoch := 0; epoch < cfg.Epochs; epoch++ {
		Shuffle(trainS)
		batches := MakeBatches(trainS, cfg.Batch, true)
		var lossSum float64
		validBatches := 0

		for _, b := range batches {
			cache, logits := Forward(p, b, cfg.Dim, cfg.Win, cfg.Hid, true, cfg.DropProb)
			lo := XentLoss(logits, b.Targets, b.Mask)
			
			// Check for NaN/Inf gradients
			if math.IsNaN(lo.Loss) || math.IsInf(lo.Loss, 0) {
				fmt.Printf("   Warning: Skipping batch with NaN/Inf loss at epoch %d\n", epoch)
				continue
			}
			
			gr := Backward(p, cache, lo.DLogits, cfg.Dim, cfg.Win, cfg.Hid, vocab, b.Mask, cfg.L2)
			
			// Check gradient norms
			hasNanGrad := false
			for i := range gr.Embed {
				for j := range gr.Embed[i] {
					if math.IsNaN(float64(gr.Embed[i][j])) || math.IsInf(float64(gr.Embed[i][j]), 0) {
						hasNanGrad = true
						break
					}
				}
				if hasNanGrad {
					break
				}
			}
			
			if hasNanGrad {
				fmt.Printf("   Warning: Skipping batch with NaN/Inf gradients at epoch %d\n", epoch)
				continue
			}
			
			applyAdam(p, moms, gr, cfg.Adam)
			lossSum += lo.Loss
			validBatches++
		}

		if validBatches == 0 {
			fmt.Printf("   Error: No valid batches in epoch %d\n", epoch)
			break
		}

		// Evaluate on validation set
		vl := EvalLoss(p, valS, cfg, vocab)
		trainLoss := lossSum / float64(validBatches)
		perplexity := math.Exp(vl)

		// Save metrics
		metrics := EpochMetrics{
			Epoch:     epoch,
			TrainLoss: trainLoss,
			ValLoss:   vl,
			Perplexity: perplexity,
		}
		allMetrics = append(allMetrics, metrics)

		// Track best model
		if vl < bestValLoss {
			bestValLoss = vl
			// Deep copy the model parameters
			bestModel = &Params{
				Embed: copyMatrix(p.Embed),
				W1:    copyMatrix(p.W1),
				B1:    copyVector(p.B1),
				W2:    copyMatrix(p.W2),
				B2:    copyVector(p.B2),
			}
		}

		// Print progress
		if epoch%5 == 0 || epoch == cfg.Epochs-1 {
			fmt.Printf("   Epoch %d: train=%.4f val=%.4f ppl=%.2f%s\n", 
				epoch, trainLoss, vl, perplexity,
				func() string {
					if vl == bestValLoss {
						return " [best]"
					}
					return ""
				}())
		}

		// Early stopping check
		if len(allMetrics) > 10 {
			recentMetrics := allMetrics[len(allMetrics)-5:]
			improving := false
			for i := 1; i < len(recentMetrics); i++ {
				if recentMetrics[i].ValLoss < recentMetrics[i-1].ValLoss*0.999 {
					improving = true
					break
				}
			}
			if !improving {
				fmt.Printf("   Early stopping: no improvement in validation loss\n")
				break
			}
		}
	}

	// Save metrics to file
	metricsData := Metrics{Epochs: allMetrics}
	metricsPath := filepath.Join(outputDir, "metrics.json")
	if err := saveMetricsJSON(metricsPath, metricsData); err != nil {
		fmt.Printf("   Warning: Failed to save metrics: %v\n", err)
	} else {
		fmt.Printf("   📊 Metrics saved to: %s\n", metricsPath)
	}

	// Use best model if available, otherwise current model
	if bestModel != nil {
		fmt.Printf("   Using best model from validation\n")
		return bestModel, bestValLoss
	}

	final := EvalLoss(p, valS, cfg, vocab)
	return p, final
}

// Helper functions for deep copying model parameters
func copyMatrix(src [][]float32) [][]float32 {
	dst := make([][]float32, len(src))
	for i := range src {
		dst[i] = make([]float32, len(src[i]))
		copy(dst[i], src[i])
	}
	return dst
}

func copyVector(src []float32) []float32 {
	dst := make([]float32, len(src))
	copy(dst, src)
	return dst
}

func saveMetricsJSON(path string, metrics Metrics) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	encoder := json.NewEncoder(f)
	encoder.SetIndent("", "  ")
	return encoder.Encode(metrics)
}
