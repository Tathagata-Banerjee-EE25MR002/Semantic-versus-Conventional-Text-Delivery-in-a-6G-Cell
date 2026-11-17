"""
word_huffman.py -- WORD-level Huffman source code (the second conventional source coder of Task 1).

The character-level Huffman code of Exercise 1 spends about 4.5 bit per character, i.e. roughly 350 bits
per sentence.  A word-level code spends about log2(1/p(word)) bits per word, i.e. roughly the word
entropy (about 10 bit/word, 150 bits per sentence) -- about half as many bits.  The price: the code
only knows the words of the vocabulary (knowledge base); every other word becomes <unk> and is lost.

The class has the SAME interface as source_coding.CharHuffman (encode(text) -> bits, decode(bits) ->
text, .stats()), so link.TxFrames / link.run, the convolutional code, the modem and the channel are
re-used unchanged.  As in Exercise 1 the code is static (built once from the training statistics, add-one
smoothed so that every vocabulary entry has a codeword) and an <eos> symbol terminates the sentence.
A single bit error desynchronises the decoder exactly as for the character code (cliff effect).
"""
from collections import Counter
import numpy as np
from source_coding import CharHuffman
from tokens import EOS, PAD, UNK


class WordHuffman(CharHuffman):
    def __init__(self, vocab, train_sentences):
        self.vocab = vocab
        freq = Counter()
        for s in train_sentences:
            freq.update(vocab.encode(s, add_eos=True))        # counts of words, <unk> (rare words) and <eos>
        for i in range(1, len(vocab)):                         # add-one smoothing (skip PAD)
            freq[i] += 1
        self.freq = freq
        self._build_tree()                                     # inherited: classical Huffman tree

    def encode(self, text):
        ids = self.vocab.encode(text, add_eos=True)
        return np.concatenate([self.code_arr[i] for i in ids])

    def decode(self, bits):
        """Walk the tree; stop at <eos>; <unk> is omitted from the text."""
        out, node = [], 0
        child, leaf = self.child, self.leaf
        for b in bits:
            node = child[node][int(b)]
            s = leaf[node]
            if s is not None:
                if s == EOS:
                    break
                out.append(s)
                node = 0
        return self.vocab.decode(out)


if __name__ == "__main__":
    from corpus import load_corpus
    from tokens import Vocab
    tr, te = load_corpus()
    v = Vocab(tr)
    h = WordHuffman(v, tr)
    print(h.stats())
    ok = [h.decode(h.encode(s)) == " ".join(w for w in v.decode(v.encode(s)).split()) for s in te[:200]]
    bits = np.mean([len(h.encode(s)) for s in te[:500]])
    print("lossless on in-vocabulary part:", all(ok), "| average bits per sentence:", round(float(bits), 1))
