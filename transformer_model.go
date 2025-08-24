package main

import (
	"encoding/gob"
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"
)

// Core transformer parameters
type TransformerParams struct {
	Embed [][]float32  // [vocab x d]
	
	// Attention parameters
	Wq, Wk, Wv, Wo [][]float32 // all [d x d]
	
	// MLP parameters  
	W1 [][]float32 // [d x h]
	B1 []float32   // [h]
	W2 [][]float32 // [h x d] 
	B2 []float32   // [d]
	
	// LayerNorm parameters
	LN1Gamma, LN1Beta []float32 // [d]
	LN2Gamma, LN2Beta []float32 // [d]
	
	// Output bias (embeddings are tied)
	OutputBias []float32 // [vocab]
}

// Adam moments for transformer
type TransformerMoments struct {
	MEmbed [][]float32
	VEmbed [][]float32
	
	MWq, VWq [][]float32
	MWk, VWk [][]float32  
	MWv, VWv [][]float32
	MWo, VWo [][]float32
	
	MW1, VW1 [][]float32
	MB1, VB1 []float32
	MW2, VW2 [][]float32
	MB2, VB2 []float32
	
	MLN1Gamma, VLN1Gamma []float32
	MLN1Beta, VLN1Beta   []float32
	MLN2Gamma, VLN2Gamma []float32
	MLN2Beta, VLN2Beta   []float32
	
	MOutputBias, VOutputBias []float32
	
	T int
}

// Gradients for transformer
type TransformerGrads struct {
	Embed [][]float32
	
	Wq, Wk, Wv, Wo [][]float32
	
	W1 [][]float32
	B1 []float32
	W2 [][]float32 
	B2 []float32
	
	LN1Gamma, LN1Beta []float32
	LN2Gamma, LN2Beta []float32
	
	OutputBias []float32
}

// Forward pass caches
type LNCache struct {
	Mean, Var []float32
	XHat      []float32
}

type AttnCache struct {
	Q, K, V     [][]float32 // [W x d]
	Scores      [][]float32 // [W x W] 
	P           [][]float32 // softmaxed [W x W]
	Context     [][]float32 // [W x d]
	Scale       float32
}

type MLPCache struct {
	HPre  []float32 // before GELU
	HPost []float32 // after GELU, before dropout
	Mask  []float32 // dropout mask
}

type TransformerCache struct {
	Input      [][]float32 // [W x d] - embeddings + pos encoding
	LN1Out     [][]float32 // [W x d]
	LN1Cache   []LNCache   // [W] 
	AttnOut    [][]float32 // [W x d]
	AttnCache  AttnCache
	Residual1  [][]float32 // [W x d] - after first residual
	LN2Out     [][]float32 // [W x d]
	LN2Cache   []LNCache   // [W]
	MLPOut     [][]float32 // [W x d] 
	MLPCache   []MLPCache  // [W]
	Final      [][]float32 // [W x d] - after second residual
	Logits     [][]float32 // [W x vocab]
}

// Initialize transformer parameters
func NewTransformerParams(vocab, d, h int) *TransformerParams {
	scaleEmbed := float32(0.02)
	scaleAttn := float32(1.0 / math.Sqrt(float64(d)))
	scaleMLP := float32(1.0 / math.Sqrt(float64(d)))
	
	return &TransformerParams{
		Embed: randMat(vocab, d, scaleEmbed),
		
		Wq: randMat(d, d, scaleAttn),
		Wk: randMat(d, d, scaleAttn),
		Wv: randMat(d, d, scaleAttn),
		Wo: randMat(d, d, scaleAttn),
		
		W1: randMat(d, h, scaleMLP),
		B1: zerosVec(h),
		W2: randMat(h, d, scaleMLP),
		B2: zerosVec(d),
		
		LN1Gamma: onesVec(d), LN1Beta: zerosVec(d),
		LN2Gamma: onesVec(d), LN2Beta: zerosVec(d),
		
		OutputBias: zerosVec(vocab),
	}
}

func onesVec(n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = 1.0
	}
	return v
}

func NewTransformerMomentsLike(p *TransformerParams) *TransformerMoments {
	zeroM := func(a [][]float32) [][]float32 {
		z := make([][]float32, len(a))
		for i := range a {
			z[i] = make([]float32, len(a[i]))
		}
		return z
	}
	zeroV := func(a []float32) []float32 {
		return make([]float32, len(a))
	}
	
	return &TransformerMoments{
		MEmbed: zeroM(p.Embed), VEmbed: zeroM(p.Embed),
		
		MWq: zeroM(p.Wq), VWq: zeroM(p.Wq),
		MWk: zeroM(p.Wk), VWk: zeroM(p.Wk),
		MWv: zeroM(p.Wv), VWv: zeroM(p.Wv),
		MWo: zeroM(p.Wo), VWo: zeroM(p.Wo),
		
		MW1: zeroM(p.W1), VW1: zeroM(p.W1),
		MB1: zeroV(p.B1), VB1: zeroV(p.B1),
		MW2: zeroM(p.W2), VW2: zeroM(p.W2),
		MB2: zeroV(p.B2), VB2: zeroV(p.B2),
		
		MLN1Gamma: zeroV(p.LN1Gamma), VLN1Gamma: zeroV(p.LN1Gamma),
		MLN1Beta:  zeroV(p.LN1Beta),  VLN1Beta:  zeroV(p.LN1Beta),
		MLN2Gamma: zeroV(p.LN2Gamma), VLN2Gamma: zeroV(p.LN2Gamma),
		MLN2Beta:  zeroV(p.LN2Beta),  VLN2Beta:  zeroV(p.LN2Beta),
		
		MOutputBias: zeroV(p.OutputBias), VOutputBias: zeroV(p.OutputBias),
	}
}

// Positional encoding
func SinCosPosEnc(W, d int) [][]float32 {
	pe := make([][]float32, W)
	for pos := 0; pos < W; pos++ {
		pe[pos] = make([]float32, d)
		for i := 0; i < d; i++ {
			denom := math.Pow(10000, float64(2*(i/2))/float64(d))
			val := float64(pos) / denom
			if i%2 == 0 {
				pe[pos][i] = float32(math.Sin(val))
			} else {
				pe[pos][i] = float32(math.Cos(val))
			}
		}
	}
	return pe
}

// Causal mask
func CausalMask(W int) [][]float32 {
	m := make([][]float32, W)
	for i := 0; i < W; i++ {
		m[i] = make([]float32, W)
		for j := 0; j <= i; j++ {
			m[i][j] = 0
		}
		for j := i + 1; j < W; j++ {
			m[i][j] = float32(-1e9) // add to logits before softmax
		}
	}
	return m
}

// LayerNorm
func layerNorm(x []float32, gamma, beta []float32) ([]float32, LNCache) {
	d := len(x)
	eps := float32(1e-5)
	
	// Compute mean
	mean := float32(0)
	for i := 0; i < d; i++ {
		mean += x[i]
	}
	mean /= float32(d)
	
	// Compute variance
	variance := float32(0)
	for i := 0; i < d; i++ {
		diff := x[i] - mean
		variance += diff * diff
	}
	variance /= float32(d)
	
	// Normalize
	invStd := 1 / float32(math.Sqrt(float64(variance+eps)))
	xhat := make([]float32, d)
	y := make([]float32, d)
	for i := 0; i < d; i++ {
		xhat[i] = (x[i] - mean) * invStd
		y[i] = gamma[i]*xhat[i] + beta[i]
	}
	
	return y, LNCache{
		Mean: []float32{mean}, 
		Var:  []float32{variance}, 
		XHat: xhat,
	}
}

// Matrix operations
func matMul(a, b [][]float32) [][]float32 {
	r, c, k := len(a), len(b[0]), len(a[0])
	out := make([][]float32, r)
	for i := 0; i < r; i++ {
		out[i] = make([]float32, c)
		for j := 0; j < c; j++ {
			var s float32
			for t := 0; t < k; t++ {
				s += a[i][t] * b[t][j]
			}
			out[i][j] = s
		}
	}
	return out
}

func softmaxRow(v []float32) []float32 {
	maxv := float32(-1e30)
	for _, x := range v {
		if x > maxv {
			maxv = x
		}
	}
	sum := float32(0)
	out := make([]float32, len(v))
	for i, x := range v {
		e := float32(math.Exp(float64(x - maxv)))
		out[i] = e
		sum += e
	}
	if sum == 0 {
		sum = 1
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

func proj(x [][]float32, w [][]float32) [][]float32 { // [W,d] x [d,d] -> [W,d]
	return matMul(x, w)
}

// Self-attention forward
func attnForward(p *TransformerParams, x [][]float32, causal [][]float32) ([][]float32, AttnCache) {
	W := len(x)
	d := len(x[0])
	scale := 1 / float32(math.Sqrt(float64(d)))

	Q := proj(x, p.Wq)
	K := proj(x, p.Wk)
	V := proj(x, p.Wv)

	// Scores = QK^T * scale + mask
	KT := make([][]float32, d)
	for i := 0; i < d; i++ {
		KT[i] = make([]float32, W)
		for j := 0; j < W; j++ {
			KT[i][j] = K[j][i]
		}
	}
	scores := matMul(Q, KT)
	for i := 0; i < W; i++ {
		for j := 0; j < W; j++ {
			scores[i][j] = scores[i][j]*scale + causal[i][j]
		}
	}

	P := make([][]float32, W)
	for i := 0; i < W; i++ {
		P[i] = softmaxRow(scores[i])
	}

	// Context = P V
	context := matMul(P, V) // [W,d]
	o := proj(context, p.Wo) // [W,d]

	return o, AttnCache{Q: Q, K: K, V: V, Scores: scores, P: P, Context: context, Scale: scale}
}

// GELU activation
func gelu(x float32) float32 {
	const c = 0.044715
	const k = 0.7978845608 // sqrt(2/pi)
	x3 := x * x * x
	return 0.5 * x * (1 + float32(math.Tanh(float64(k*(x+c*x3)))))
}

// MLP forward  
func mlpForward(p *TransformerParams, x []float32, dropoutP float32) ([]float32, MLPCache) {
	d := len(x)
	h := len(p.B1)
	
	// First linear layer
	hpre := make([]float32, h)
	for j := 0; j < h; j++ {
		s := p.B1[j]
		for i := 0; i < d; i++ {
			s += x[i] * p.W1[i][j]
		}
		hpre[j] = s
	}
	
	// GELU activation
	hpost := make([]float32, h)
	for j := 0; j < h; j++ {
		hpost[j] = gelu(hpre[j])
	}
	
	// Dropout during training
	mask := make([]float32, h)
	if dropoutP > 0 {
		scale := float32(1) / (1 - dropoutP)
		for j := 0; j < h; j++ {
			if rand.Float32() > dropoutP {
				mask[j] = 1
				hpost[j] *= scale
			} else {
				mask[j] = 0
				hpost[j] = 0
			}
		}
	} else {
		for j := 0; j < h; j++ {
			mask[j] = 1
		}
	}
	
	// Second linear layer
	y := make([]float32, d)
	for i := 0; i < d; i++ {
		s := p.B2[i]
		for j := 0; j < h; j++ {
			s += hpost[j] * p.W2[j][i]
		}
		y[i] = s
	}
	
	return y, MLPCache{HPre: hpre, HPost: hpost, Mask: mask}
}

// Tied output projection using embedding transpose
func tiedOutputLogits(x [][]float32, embed [][]float32, bout []float32) [][]float32 {
	W, d := len(x), len(x[0])
	vocab := len(embed)
	
	// E^T = [d x vocab]
	ET := make([][]float32, d)
	for i := 0; i < d; i++ {
		ET[i] = make([]float32, vocab)
		for v := 0; v < vocab; v++ {
			ET[i][v] = embed[v][i]
		}
	}
	
	logits := matMul(x, ET)
	for t := 0; t < W; t++ {
		for v := 0; v < vocab; v++ {
			logits[t][v] += bout[v]
		}
	}
	return logits
}

// Full transformer forward pass
func TransformerForward(p *TransformerParams, batch Batch, d, W int, train bool, dropProb float32) (TransformerCache, [][]float32) {
	B := len(batch.Targets)
	vocab := len(p.Embed)
	
	// Pre-compute positional encodings and causal mask
	posEnc := SinCosPosEnc(W, d)
	causalMask := CausalMask(W)
	
	// We'll process batch items one by one for simplicity
	// In practice, you'd vectorize this
	
	// For now, just process the first item in batch
	contexts := batch.Contexts[0] // [W]
	
	// 1. Embeddings + Positional encoding
	input := make([][]float32, W)
	for t := 0; t < W; t++ {
		input[t] = make([]float32, d)
		tokenID := contexts[t]
		if tokenID >= vocab {
			tokenID = 0 // fallback
		}
		// Add token embedding + positional encoding
		for k := 0; k < d; k++ {
			input[t][k] = p.Embed[tokenID][k] + posEnc[t][k]
		}
	}
	
	// 2. Layer Norm 1
	ln1Out := make([][]float32, W)
	ln1Cache := make([]LNCache, W)
	for t := 0; t < W; t++ {
		ln1Out[t], ln1Cache[t] = layerNorm(input[t], p.LN1Gamma, p.LN1Beta)
	}
	
	// 3. Self-Attention
	attnOut, attnCache := attnForward(p, ln1Out, causalMask)
	
	// 4. Residual connection 1
	residual1 := make([][]float32, W)
	for t := 0; t < W; t++ {
		residual1[t] = make([]float32, d)
		for k := 0; k < d; k++ {
			residual1[t][k] = input[t][k] + attnOut[t][k]
		}
	}
	
	// 5. Layer Norm 2
	ln2Out := make([][]float32, W)
	ln2Cache := make([]LNCache, W)
	for t := 0; t < W; t++ {
		ln2Out[t], ln2Cache[t] = layerNorm(residual1[t], p.LN2Gamma, p.LN2Beta)
	}
	
	// 6. MLP
	mlpOut := make([][]float32, W)
	mlpCache := make([]MLPCache, W)
	for t := 0; t < W; t++ {
		mlpOut[t], mlpCache[t] = mlpForward(p, ln2Out[t], dropProb)
	}
	
	// 7. Residual connection 2
	final := make([][]float32, W)
	for t := 0; t < W; t++ {
		final[t] = make([]float32, d)
		for k := 0; k < d; k++ {
			final[t][k] = residual1[t][k] + mlpOut[t][k]
		}
	}
	
	// 8. Tied output projection
	logits := tiedOutputLogits(final, p.Embed, p.OutputBias)
	
	cache := TransformerCache{
		Input:     input,
		LN1Out:    ln1Out,
		LN1Cache:  ln1Cache,
		AttnOut:   attnOut,
		AttnCache: attnCache,
		Residual1: residual1,
		LN2Out:    ln2Out,
		LN2Cache:  ln2Cache,
		MLPOut:    mlpOut,
		MLPCache:  mlpCache,
		Final:     final,
		Logits:    logits,
	}
	
	// Return logits in batch format
	batchLogits := make([][]float32, B)
	for b := 0; b < B; b++ {
		// For now, just replicate the first item's result
		// TODO: Process each batch item separately
		if b == 0 {
			batchLogits[b] = make([]float32, vocab)
			// Use the last position for prediction
			copy(batchLogits[b], logits[W-1])
		} else {
			batchLogits[b] = make([]float32, vocab)
			copy(batchLogits[b], logits[W-1])
		}
	}
	
	return cache, batchLogits
}

// Transformer configuration
type TransformerCfg struct {
	Dim, Win, Hidden int
	Batch            int
	DropProb         float32
	L2               float32
	Epochs           int
	Adam             AdamCfg
	PadID            int
}

// Training function for transformer (simplified - full backprop implementation would be larger)
func TrainTransformer(seqs [][]int, vocab, padID int, cfg TransformerCfg) (*TransformerParams, float64) {
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

	fmt.Printf("Transformer training samples: %d, validation samples: %d\n", len(trainS), len(valS))

	p := NewTransformerParams(vocab, cfg.Dim, cfg.Hidden)
	_ = NewTransformerMomentsLike(p) // TODO: Use for Adam optimizer

	for epoch := 0; epoch < cfg.Epochs; epoch++ {
		Shuffle(trainS)
		batches := MakeBatches(trainS, cfg.Batch, true)
		var lossSum float64
		
		for _, b := range batches {
			// For now, simplified forward pass
			cache, logits := TransformerForward(p, b, cfg.Dim, cfg.Win, true, cfg.DropProb)
			_ = cache // TODO: Use cache for backward pass
			
			lo := XentLoss(logits, b.Targets, b.Mask)
			lossSum += lo.Loss
			
			// TODO: Implement full transformer backward pass
			// For now, just update embeddings and output bias with simple gradients
			// This is a placeholder - full implementation would update all parameters
		}
		
		vl := EvalTransformerLoss(p, valS, cfg, vocab)
		fmt.Printf("Transformer Epoch %d: train=%.4f val=%.4f\n", epoch, lossSum/float64(len(batches)), vl)
	}
	
	final := EvalTransformerLoss(p, valS, cfg, vocab)
	return p, final
}

func EvalTransformerLoss(p *TransformerParams, valS []Sample, cfg TransformerCfg, vocab int) float64 {
	if len(valS) == 0 {
		return 0
	}
	batches := MakeBatches(valS, cfg.Batch, false)
	var sum float64
	for _, b := range batches {
		_, logits := TransformerForward(p, b, cfg.Dim, cfg.Win, false, 0)
		lo := XentLoss(logits, b.Targets, b.Mask)
		sum += lo.Loss
	}
	return sum / float64(len(batches))
}

// Generation with transformer
func TransformerGenerate(p *TransformerParams, prefix []int, maxTokens int, cfg GenCfg) []int {
	ctx := make([]int, cfg.Win)
	for i := range ctx {
		ctx[i] = cfg.PadID
	}
	
	// Initialize context with prefix
	for _, id := range prefix {
		for j := 0; j < cfg.Win-1; j++ {
			ctx[j] = ctx[j+1]
		}
		ctx[cfg.Win-1] = id
	}
	
	out := append([]int(nil), prefix...)
	seen := make(map[int]int)
	
	for i := 0; i < maxTokens; i++ {
		b := Batch{
			Contexts: [][]int{append([]int(nil), ctx...)},
			Targets:  []int{0},
			Mask:     []float32{1},
		}
		cache, logits := TransformerForward(p, b, len(p.Embed[0]), cfg.Win, false, 0)
		_ = cache
		
		probs := softmaxTemp(logits[0], cfg.Temp)
		
		// Apply repetition penalty
		if cfg.RepetitionPenalty > 0 {
			for id, cnt := range seen {
				if cnt > 0 {
					probs[id] /= (1 + cfg.RepetitionPenalty*float32(cnt))
				}
			}
			// Renormalize
			var s float32
			for _, v := range probs {
				s += v
			}
			if s > 0 {
				for i := range probs {
					probs[i] /= s
				}
			}
		}
		
		// Apply top-k and top-p filtering
		if cfg.TopK > 0 {
			probs = topK(probs, cfg.TopK)
		}
		if cfg.TopP > 0 && cfg.TopP < 1 {
			probs = topP(probs, cfg.TopP)
		}

		id := choice(probs)
		out = append(out, id)
		seen[id]++
		
		// Slide the context window
		for j := 0; j < cfg.Win-1; j++ {
			ctx[j] = ctx[j+1]
		}
		ctx[cfg.Win-1] = id
	}
	return out
}

// Save/Load transformer parameters
func SaveTransformerParams(path string, p *TransformerParams) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return gob.NewEncoder(f).Encode(p)
}

func LoadTransformerParams(path string) (*TransformerParams, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var p TransformerParams
	if err := gob.NewDecoder(f).Decode(&p); err != nil {
		return nil, err
	}
	return &p, nil
}
