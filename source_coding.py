"""
source_coding.py -- character-level Huffman source encoder / decoder.

Design choices (all explained because the problem asks for a *conventional*
bit-level chain in the sense of Shannon & Weaver):

* The Huffman code is built ONCE from the character statistics of the training
  sentences and is assumed known at both ends (it is a static codebook, like the
  one in a standard, so no side-information has to be transmitted).
* Add-one smoothing over all 95 printable ASCII characters guarantees that every
  test character has a codeword even if it never appeared in training.
* A special end-of-sentence symbol <EOS> is part of the alphabet.  It lets the
  decoder know where the sentence ends, so zero-padding of the frame (needed to
  batch frames of different length in the vectorised Viterbi decoder) is ignored.
* Huffman codes are *variable length and prefix-free*: a single flipped bit makes
  the decoder lose codeword synchronisation, so the rest of the sentence is
  decoded as garbage.  This error propagation is one of the two causes of the
  "cliff effect" analysed in Task 1(c).
"""
import heapq
from collections import Counter
import numpy as np

EOS = "\x00"                      # end-of-sentence marker (not a printable char)


class CharHuffman:
    def __init__(self, train_sentences):
        freq = Counter()
        for s in train_sentences:
            freq.update(s)
        for c in range(32, 127):                  # add-one smoothing
            freq[chr(c)] += 1
        freq[EOS] = len(train_sentences)          # one EOS per sentence
        self.freq = freq
        self._build_tree()

    # ---------------------------------------------------------------- build
    def _build_tree(self):
        """Classical Huffman: repeatedly merge the two least-frequent nodes."""
        heap, cnt = [], 0
        for sym, f in self.freq.items():
            heap.append((f, cnt, sym)); cnt += 1          # cnt = deterministic tie-break
        heapq.heapify(heap)
        children = {}                                      # node id -> (left, right)
        while len(heap) > 1:
            f1, _, a = heapq.heappop(heap)
            f2, _, b = heapq.heappop(heap)
            node = ("node", cnt)
            children[node] = (a, b)
            heapq.heappush(heap, (f1 + f2, cnt, node)); cnt += 1
        root = heap[0][2]
        self.codebook = {}                                 # symbol -> '0101...'
        # flat arrays for a fast table-free decoder: child[node][bit], leaf symbol
        self.child, self.leaf = [[-1, -1]], [None]
        stack = [(root, 0, "")]
        while stack:
            node, idx, prefix = stack.pop()
            if isinstance(node, tuple) and node[0] == "node":
                l, r = children[node]
                for bit, sub in ((0, l), (1, r)):
                    self.child.append([-1, -1]); self.leaf.append(None)
                    sub_idx = len(self.child) - 1
                    self.child[idx][bit] = sub_idx
                    stack.append((sub, sub_idx, prefix + str(bit)))
            else:
                self.leaf[idx] = node
                self.codebook[node] = prefix
        self.code_arr = {s: np.fromiter((int(b) for b in c), np.uint8)
                         for s, c in self.codebook.items()}

    # ---------------------------------------------------------------- API
    def encode(self, text):
        """text -> np.uint8 bit array (sentence followed by <EOS>)."""
        parts = [self.code_arr[c] for c in text] + [self.code_arr[EOS]]
        return np.concatenate(parts)

    def decode(self, bits):
        """bit array -> text.  Stops at the first <EOS> (or when bits run out)."""
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
        return "".join(out)

    # ---------------------------------------------------------------- stats
    def stats(self):
        """Entropy and average codeword length per character (for the paper)."""
        tot = sum(self.freq.values())
        H = -sum(f / tot * np.log2(f / tot) for f in self.freq.values())
        L = sum(f / tot * len(self.codebook[s]) for s, f in self.freq.items())
        return dict(entropy_bits=float(H), avg_len_bits=float(L), n_symbols=len(self.freq))


if __name__ == "__main__":
    from corpus import load_corpus
    tr, te = load_corpus()
    h = CharHuffman(tr)
    print(h.stats())
    b = h.encode(te[0]); print(len(te[0]), len(b), h.decode(b) == te[0])
