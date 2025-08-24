package main

import (
	"encoding/gob"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"time"
)

type Params struct {
	Embed [][]float32
	W1    [][]float32
	B1    []float32
	W2    [][]float32
	B2    []float32
}

type Moments struct {
	MEmbed [][]float32
	VEmbed [][]float32
	MW1    [][]float32
	VW1    [][]float32
	MB1    []float32
	VB1    []float32
	MW2    [][]float32
	VW2    [][]float32
	MB2    []float32
	VB2    []float32
	T      int
}

type Grads struct {
	Embed [][]float32
	W1    [][]float32
	B1    []float32
	W2    [][]float32
	B2    []float32
}

type ForwardCache struct {
	Contexts [][]int
	EmbCat   [][]float32
	HPre     [][]float32
	HAct     [][]float32
	DropMask [][]float32
	Logits   [][]float32
}

type Sample struct {
	Context []int
	Target  int
}

type Batch struct {
	Contexts [][]int
	Targets  []int
	Mask     []float32
}

type LossOut struct {
	Loss    float64
	Prob    [][]float32
	DLogits [][]float32
}

type AdamCfg struct {
	LR, Beta1, Beta2, Eps float32
	Clip                  float32
}

type TrainCfg struct {
	Dim, Win, Hid int
	Batch         int
	DropProb      float32
	L2            float32
	Epochs        int
	Adam          AdamCfg
	PadID         int
}

type GenCfg struct {
	Win               int
	Temp              float32
	TopK              int
	TopP              float32
	RepetitionPenalty float32
	PadID             int
}

// Initialization functions
func randMat(rows, cols int, scale float32) [][]float32 {
	m := make([][]float32, rows)
	for i := range m {
		m[i] = make([]float32, cols)
		for j := range m[i] {
			m[i][j] = (rand.Float32()*2 - 1) * scale
		}
	}
	return m
}

func zerosMat(rows, cols int) [][]float32 {
	m := make([][]float32, rows)
	for i := range m {
		m[i] = make([]float32, cols)
	}
	return m
}

func zerosVec(n int) []float32 {
	return make([]float32, n)
}

func NewParams(vocab, d, w, h int) *Params {
	scaleE := float32(0.05)
	scale1 := float32(1.0 / math.Sqrt(float64(d*w)))
	scale2 := float32(1.0 / math.Sqrt(float64(h)))
	return &Params{
		Embed: randMat(vocab, d, scaleE),
		W1:    randMat(d*w, h, scale1),
		B1:    zerosVec(h),
		W2:    randMat(h, vocab, scale2),
		B2:    zerosVec(vocab),
	}
}

func NewMomentsLike(p *Params) *Moments {
	zeroM := func(a [][]float32) [][]float32 {
		z := make([][]float32, len(a))
		for i := range a {
			z[i] = make([]float32, len(a[i]))
		}
		return z
	}
	return &Moments{
		MEmbed: zeroM(p.Embed), VEmbed: zeroM(p.Embed),
		MW1: zeroM(p.W1), VW1: zeroM(p.W1),
		MB1: make([]float32, len(p.B1)), VB1: make([]float32, len(p.B1)),
		MW2: zeroM(p.W2), VW2: zeroM(p.W2),
		MB2: make([]float32, len(p.B2)), VB2: make([]float32, len(p.B2)),
	}
}

// Dataset windows and batches
func MakeWindowSamples(seq []int, W int, padID int) []Sample {
	out := make([]Sample, 0, len(seq))
	ctx := make([]int, W)
	for i := range ctx {
		ctx[i] = padID
	}
	for i := 0; i < len(seq); i++ {
		copyCtx := make([]int, W)
		copy(copyCtx, ctx)
		out = append(out, Sample{Context: copyCtx, Target: seq[i]})
		// slide window
		for j := 0; j < W-1; j++ {
			ctx[j] = ctx[j+1]
		}
		ctx[W-1] = seq[i]
	}
	return out
}

func Shuffle[T any](a []T) {
	rand.Shuffle(len(a), func(i, j int) { a[i], a[j] = a[j], a[i] })
}

func MakeBatches(samples []Sample, batchSize int, dropLast bool) []Batch {
	Shuffle(samples)
	var batches []Batch
	for i := 0; i < len(samples); i += batchSize {
		j := i + batchSize
		if j > len(samples) {
			if dropLast {
				break
			}
			j = len(samples)
		}
		b := Batch{
			Contexts: make([][]int, j-i),
			Targets:  make([]int, j-i),
			Mask:     make([]float32, j-i),
		}
		for k := i; k < j; k++ {
			idx := k - i
			b.Contexts[idx] = samples[k].Context
			b.Targets[idx] = samples[k].Target
			b.Mask[idx] = 1
		}
		batches = append(batches, b)
	}
	return batches
}

// Math utilities
func matVec(m [][]float32, v []float32) []float32 {
	out := make([]float32, len(m[0]))
	for i := range m {
		mi := m[i]
		vi := v[i]
		for j := range out {
			out[j] += mi[j] * vi
		}
	}
	return out
}

func addVec(a, b []float32) []float32 {
	out := make([]float32, len(a))
	for i := range a {
		out[i] = a[i] + b[i]
	}
	return out
}

func relu(v []float32) ([]float32, []float32) {
	out := make([]float32, len(v))
	mask := make([]float32, len(v))
	for i := range v {
		if v[i] > 0 {
			out[i] = v[i]
			mask[i] = 1
		}
	}
	return out, mask
}

func dropout(v []float32, p float32) ([]float32, []float32) {
	if p <= 0 {
		return append([]float32(nil), v...), make([]float32, len(v))
	}
	scale := float32(1.0) / (1 - p)
	mask := make([]float32, len(v))
	out := make([]float32, len(v))
	for i := range v {
		if rand.Float32() > p {
			mask[i] = 1
			out[i] = v[i] * scale
		} else {
			mask[i] = 0
		}
	}
	return out, mask
}

// Forward pass
func Forward(p *Params, b Batch, d, W, H int, train bool, dropProb float32) (ForwardCache, [][]float32) {
	B := len(b.Targets)
	embCat := make([][]float32, B)
	hpre := make([][]float32, B)
	hact := make([][]float32, B)
	dmask := make([][]float32, B)
	logits := make([][]float32, B)

	for i := 0; i < B; i++ {
		// Concatenate W embeddings
		x := make([]float32, W*d)
		for w := 0; w < W; w++ {
			id := b.Contexts[i][w]
			copy(x[w*d:(w+1)*d], p.Embed[id])
		}
		embCat[i] = x

		// First linear layer
		h := matVec(p.W1, x)
		h = addVec(h, p.B1)
		hpre[i] = h

		// ReLU activation
		r, _ := relu(h)

		// Dropout during training
		if train {
			r, dm := dropout(r, dropProb)
			dmask[i] = dm
			hact[i] = r
		} else {
			hact[i] = r
			dmask[i] = make([]float32, len(r))
			for k := range dmask[i] {
				dmask[i][k] = 1
			}
		}

		// Output layer
		o := matVec(p.W2, hact[i])
		o = addVec(o, p.B2)
		logits[i] = o
	}

	cache := ForwardCache{
		Contexts: b.Contexts,
		EmbCat:   embCat,
		HPre:     hpre,
		HAct:     hact,
		DropMask: dmask,
		Logits:   logits,
	}
	return cache, logits
}

// Loss and softmax
func rowSoftmax(logits []float32) []float32 {
	maxv := float32(-1e30)
	for _, v := range logits {
		if v > maxv {
			maxv = v
		}
	}
	sum := float32(0)
	out := make([]float32, len(logits))
	for i, v := range logits {
		e := float32(math.Exp(float64(v - maxv)))
		out[i] = e
		sum += e
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

func XentLoss(logits [][]float32, targets []int, mask []float32) LossOut {
	B := len(targets)
	prob := make([][]float32, B)
	dlog := make([][]float32, B)
	var total float64
	var denom float64

	for i := 0; i < B; i++ {
		if mask[i] == 0 {
			continue
		}
		p := rowSoftmax(logits[i])
		prob[i] = p
		d := make([]float32, len(p))
		for j := range d {
			d[j] = p[j]
		}
		d[targets[i]] -= 1
		dlog[i] = d
		total += -math.Log(math.Max(1e-12, float64(p[targets[i]])))
		denom += 1
	}
	if denom == 0 {
		denom = 1
	}
	return LossOut{Loss: total / denom, Prob: prob, DLogits: dlog}
}

// Backward pass
func Backward(p *Params, cache ForwardCache, dout [][]float32, d, W, H, vocab int, mask []float32, weightDecay float32) *Grads {
	B := len(dout)
	g := &Grads{
		Embed: zerosMat(len(p.Embed), d),
		W1:    zerosMat(W*d, H),
		B1:    make([]float32, H),
		W2:    zerosMat(H, vocab),
		B2:    make([]float32, vocab),
	}

	// Gradients for output layer (W2, B2) and hidden activations
	dH := make([][]float32, B)
	for i := 0; i < B; i++ {
		if mask[i] == 0 {
			continue
		}
		// dB2
		for j := 0; j < vocab; j++ {
			g.B2[j] += dout[i][j]
		}
		// dW2 and dH
		for h := 0; h < H; h++ {
			sum := float32(0)
			for j := 0; j < vocab; j++ {
				g.W2[h][j] += cache.HAct[i][h] * dout[i][j]
				sum += p.W2[h][j] * dout[i][j]
			}
			// Apply dropout mask and ReLU derivative
			dHrow := float32(cache.DropMask[i][h]) * func() float32 {
				if cache.HPre[i][h] > 0 {
					return 1
				}
				return 0
			}() * sum
			if dH[i] == nil {
				dH[i] = make([]float32, H)
			}
			dH[i][h] = dHrow
		}
	}

	// Gradients for first layer (W1, B1) and embeddings
	for i := 0; i < B; i++ {
		if mask[i] == 0 {
			continue
		}
		// dB1
		for h := 0; h < H; h++ {
			g.B1[h] += dH[i][h]
		}
		// dW1
		for r := 0; r < W*d; r++ {
			xr := cache.EmbCat[i][r]
			for h := 0; h < H; h++ {
				g.W1[r][h] += xr * dH[i][h]
			}
		}
		// dX (gradient w.r.t. concatenated embeddings)
		dX := make([]float32, W*d)
		for r := 0; r < W*d; r++ {
			sum := float32(0)
			for h := 0; h < H; h++ {
				sum += p.W1[r][h] * dH[i][h]
			}
			dX[r] = sum
		}
		// Scatter gradients to individual embeddings
		for w := 0; w < W; w++ {
			id := cache.Contexts[i][w]
			base := w * d
			for k := 0; k < d; k++ {
				g.Embed[id][k] += dX[base+k]
			}
		}
	}

	// Weight decay (L2 regularization)
	if weightDecay > 0 {
		for i := range p.W1 {
			for j := range p.W1[i] {
				g.W1[i][j] += weightDecay * p.W1[i][j]
			}
		}
		for i := range p.W2 {
			for j := range p.W2[i] {
				g.W2[i][j] += weightDecay * p.W2[i][j]
			}
		}
	}
	return g
}

// Adam optimizer
func applyAdam(p *Params, m *Moments, g *Grads, cfg AdamCfg) {
	m.T++
	b1t := float32(1) - float32(math.Pow(float64(cfg.Beta1), float64(m.T)))
	b2t := float32(1) - float32(math.Pow(float64(cfg.Beta2), float64(m.T)))

	clip := func(x float32) float32 {
		if cfg.Clip <= 0 {
			return x
		}
		if x > cfg.Clip {
			return cfg.Clip
		}
		if x < -cfg.Clip {
			return -cfg.Clip
		}
		return x
	}

	updMat := func(w, mw, vw [][]float32, gw [][]float32) {
		for i := range w {
			for j := range w[i] {
				gij := clip(gw[i][j])
				mw[i][j] = cfg.Beta1*mw[i][j] + (1-cfg.Beta1)*gij
				vw[i][j] = cfg.Beta2*vw[i][j] + (1-cfg.Beta2)*gij*gij
				mh := mw[i][j] / b1t
				vh := vw[i][j] / b2t
				w[i][j] -= cfg.LR * mh / (float32(math.Sqrt(float64(vh))) + cfg.Eps)
			}
		}
	}
	updVec := func(w, mw, vw []float32, gw []float32) {
		for i := range w {
			gi := clip(gw[i])
			mw[i] = cfg.Beta1*mw[i] + (1-cfg.Beta1)*gi
			vw[i] = cfg.Beta2*vw[i] + (1-cfg.Beta2)*gi*gi
			mh := mw[i] / b1t
			vh := vw[i] / b2t
			w[i] -= cfg.LR * mh / (float32(math.Sqrt(float64(vh))) + cfg.Eps)
		}
	}

	updMat(p.Embed, m.MEmbed, m.VEmbed, g.Embed)
	updMat(p.W1, m.MW1, m.VW1, g.W1)
	updVec(p.B1, m.MB1, m.VB1, g.B1)
	updMat(p.W2, m.MW2, m.VW2, g.W2)
	updVec(p.B2, m.MB2, m.VB2, g.B2)
}

// Training loop
func Train(seqs [][]int, vocab, padID int, cfg TrainCfg) (*Params, float64) {
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

	fmt.Printf("Training samples: %d, Validation samples: %d\n", len(trainS), len(valS))

	p := NewParams(vocab, cfg.Dim, cfg.Win, cfg.Hid)
	moms := NewMomentsLike(p)

	for epoch := 0; epoch < cfg.Epochs; epoch++ {
		Shuffle(trainS)
		batches := MakeBatches(trainS, cfg.Batch, true)
		var lossSum float64

		for _, b := range batches {
			cache, logits := Forward(p, b, cfg.Dim, cfg.Win, cfg.Hid, true, cfg.DropProb)
			lo := XentLoss(logits, b.Targets, b.Mask)
			gr := Backward(p, cache, lo.DLogits, cfg.Dim, cfg.Win, cfg.Hid, vocab, b.Mask, cfg.L2)
			applyAdam(p, moms, gr, cfg.Adam)
			lossSum += lo.Loss
		}

		vl := EvalLoss(p, valS, cfg, vocab)
		fmt.Printf("Epoch %d: train=%.4f val=%.4f\n", epoch, lossSum/float64(len(batches)), vl)
	}

	final := EvalLoss(p, valS, cfg, vocab)
	return p, final
}

func EvalLoss(p *Params, valS []Sample, cfg TrainCfg, vocab int) float64 {
	if len(valS) == 0 {
		return 0
	}
	batches := MakeBatches(valS, cfg.Batch, false)
	var sum float64
	for _, b := range batches {
		_, logits := Forward(p, b, cfg.Dim, cfg.Win, cfg.Hid, false, 0)
		lo := XentLoss(logits, b.Targets, b.Mask)
		sum += lo.Loss
	}
	return sum / float64(len(batches))
}

// Sampling utilities
func softmaxTemp(logits []float32, temp float32) []float32 {
	if temp <= 0 {
		temp = 1
	}
	scaled := make([]float32, len(logits))
	for i := range logits {
		scaled[i] = logits[i] / temp
	}
	return rowSoftmax(scaled)
}

func topK(probs []float32, k int) []float32 {
	if k <= 0 || k >= len(probs) {
		return probs
	}
	type kv struct {
		i int
		p float32
	}
	arr := make([]kv, len(probs))
	for i, p := range probs {
		arr[i] = kv{i, p}
	}
	sort.Slice(arr, func(i, j int) bool { return arr[i].p > arr[j].p })
	keep := arr[:k]
	out := make([]float32, len(probs))
	var s float32
	for _, e := range keep {
		out[e.i] = e.p
		s += e.p
	}
	if s == 0 {
		return probs
	}
	for i := range out {
		out[i] /= s
	}
	return out
}

func topP(probs []float32, pth float32) []float32 {
	type kv struct {
		i int
		p float32
	}
	arr := make([]kv, len(probs))
	for i, p := range probs {
		arr[i] = kv{i, p}
	}
	sort.Slice(arr, func(i, j int) bool { return arr[i].p > arr[j].p })
	var cum float32
	var idx int
	for idx = 0; idx < len(arr); idx++ {
		cum += arr[idx].p
		if cum >= pth {
			idx++
			break
		}
	}
	out := make([]float32, len(probs))
	var s float32
	for i := 0; i < idx; i++ {
		out[arr[i].i] = arr[i].p
		s += arr[i].p
	}
	if s == 0 {
		return probs
	}
	for i := range out {
		out[i] /= s
	}
	return out
}

func choice(probs []float32) int {
	r := rand.Float32()
	var cum float32
	for i, p := range probs {
		cum += p
		if r <= cum {
			return i
		}
	}
	return len(probs) - 1
}

// Text generation
func Generate(p *Params, prefix []int, maxTokens int, cfg GenCfg) []int {
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
		cache, logits := Forward(p, b, len(p.Embed[0]), cfg.Win, len(p.W2), false, 0)
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

// Save/Load functionality
func SaveParams(path string, p *Params) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return gob.NewEncoder(f).Encode(p)
}

func LoadParams(path string) (*Params, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var p Params
	if err := gob.NewDecoder(f).Decode(&p); err != nil {
		return nil, err
	}
	return &p, nil
}
