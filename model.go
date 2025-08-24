package main

import (
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type Embedding struct {
	weights *gorgonia.Node
	dim     int
}

func NewEmbedding(g *gorgonia.ExprGraph, vocab, dim int) *Embedding {
	w := gorgonia.NewMatrix(g,
		tensor.Float32,
		gorgonia.WithShape(vocab, dim),
		gorgonia.WithName("embed_weights"),
		gorgonia.WithInit(gorgonia.GlorotU(1.0)),
	)
	return &Embedding{weights: w, dim: dim}
}

func (e *Embedding) Lookup(g *gorgonia.ExprGraph, id *gorgonia.Node) (*gorgonia.Node, error) {
	// Create one-hot vector and multiply with embedding matrix
	vocab := e.weights.Shape()[0]
	oneHot := gorgonia.NewMatrix(g, tensor.Float32, gorgonia.WithShape(1, vocab), gorgonia.WithName("onehot"))
	
	// For now, we'll use a simpler approach - direct indexing
	// This is a workaround since Gorgonia's slicing can be tricky
	return gorgonia.Slice(e.weights, gorgonia.S(id))
}

type TinyLM struct {
	embed  *Embedding
	w1, b1 *gorgonia.Node
	w2, b2 *gorgonia.Node
	g      *gorgonia.ExprGraph
}

func NewTinyLM(g *gorgonia.ExprGraph, vocab, emb, hidden int) *TinyLM {
	return &TinyLM{
		embed: NewEmbedding(g, vocab, emb),
		w1: gorgonia.NewMatrix(g, tensor.Float32, 
			gorgonia.WithShape(emb, hidden), 
			gorgonia.WithName("w1"),
			gorgonia.WithInit(gorgonia.GlorotU(1.0))),
		b1: gorgonia.NewVector(g, tensor.Float32, 
			gorgonia.WithShape(hidden),
			gorgonia.WithName("b1")),
		w2: gorgonia.NewMatrix(g, tensor.Float32, 
			gorgonia.WithShape(hidden, vocab), 
			gorgonia.WithName("w2"),
			gorgonia.WithInit(gorgonia.GlorotU(1.0))),
		b2: gorgonia.NewVector(g, tensor.Float32, 
			gorgonia.WithShape(vocab),
			gorgonia.WithName("b2")),
		g: g,
	}
}

func (m *TinyLM) Forward(id *gorgonia.Node) (*gorgonia.Node, error) {
	// Embedding lookup
	x, err := m.embed.Lookup(m.g, id)
	if err != nil {
		return nil, err
	}
	
	// First layer: x * W1 + b1
	xw1, err := gorgonia.Mul(x, m.w1)
	if err != nil {
		return nil, err
	}
	
	h, err := gorgonia.Add(xw1, m.b1)
	if err != nil {
		return nil, err
	}
	
	// ReLU activation
	hAct, err := gorgonia.Rectify(h)
	if err != nil {
		return nil, err
	}
	
	// Output layer: h * W2 + b2
	hw2, err := gorgonia.Mul(hAct, m.w2)
	if err != nil {
		return nil, err
	}
	
	logits, err := gorgonia.Add(hw2, m.b2)
	if err != nil {
		return nil, err
	}
	
	return logits, nil
}

// GetLearnables returns all trainable parameters
func (m *TinyLM) GetLearnables() []*gorgonia.Node {
	return []*gorgonia.Node{
		m.embed.weights,
		m.w1, m.b1,
		m.w2, m.b2,
	}
}
