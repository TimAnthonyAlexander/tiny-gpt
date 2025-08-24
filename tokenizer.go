package main

import (
	"strings"
	"unicode"
)

type Tokenizer struct {
	toID   map[string]int
	toWord map[int]string
	vocabSize int
}

func NewTokenizer() *Tokenizer {
	return &Tokenizer{
		toID:   make(map[string]int),
		toWord: make(map[int]string),
	}
}

// BuildVocab creates vocabulary from text corpus (character-level)
func (t *Tokenizer) BuildVocab(corpus string, maxVocabSize int) {
	// Count character frequencies
	charCount := make(map[string]int)
	
	// Add special tokens
	charCount["<pad>"] = 1000000 // ensure special tokens get priority
	charCount["<unk>"] = 1000000
	charCount["<start>"] = 1000000
	charCount["<end>"] = 1000000
	
	// Count all characters in corpus
	for _, r := range corpus {
		char := string(r)
		if unicode.IsSpace(r) {
			char = " " // normalize all whitespace to space
		}
		charCount[char]++
	}
	
	// Sort by frequency and build vocab
	type charFreq struct {
		char string
		freq int
	}
	
	var chars []charFreq
	for char, freq := range charCount {
		chars = append(chars, charFreq{char, freq})
	}
	
	// Simple selection of most frequent characters
	vocabId := 0
	for _, cf := range chars {
		if vocabId >= maxVocabSize {
			break
		}
		t.toID[cf.char] = vocabId
		t.toWord[vocabId] = cf.char
		vocabId++
	}
	
	t.vocabSize = vocabId
}

// Encode converts text to token IDs
func (t *Tokenizer) Encode(text string) []int {
	var ids []int
	
	// Add start token
	ids = append(ids, t.toID["<start>"])
	
	for _, r := range text {
		char := string(r)
		if unicode.IsSpace(r) {
			char = " "
		}
		
		if id, exists := t.toID[char]; exists {
			ids = append(ids, id)
		} else {
			ids = append(ids, t.toID["<unk>"])
		}
	}
	
	// Add end token
	ids = append(ids, t.toID["<end>"])
	
	return ids
}

// Decode converts token IDs back to text
func (t *Tokenizer) Decode(ids []int) string {
	var result strings.Builder
	
	for _, id := range ids {
		if word, exists := t.toWord[id]; exists {
			if word != "<start>" && word != "<end>" && word != "<pad>" {
				result.WriteString(word)
			}
		}
	}
	
	return result.String()
}

// GetVocabSize returns the vocabulary size
func (t *Tokenizer) GetVocabSize() int {
	return t.vocabSize
}
