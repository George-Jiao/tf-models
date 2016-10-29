import collections
import os
import nltk
import numpy as np
from collections import Counter
from collections import defaultdict

# n-gram stuff

def bigram_counts(word_list):
  bgs = nltk.bigrams(word_list)
  fdist = nltk.FreqDist(bgs)
  d = Counter()
  for k, v in fdist.items():
    d[k] = v
  return d

def trigram_counts(word_list):
  tgs = nltk.trigrams(word_list)
  fdist = nltk.FreqDist(tgs)
  d = Counter()
  for k, v in fdist.items():
    d[k] = v
  return d

def build_continuations(counts_dict):
  total = defaultdict(int)
  distinct = defaultdict(int)
  for key in counts_dict:
    context = key[:-1]
    total[context] += counts_dict[key]
    distinct[context] += 1
  return {"total": total, "distinct": distinct}

def estimate_modkn_discounts(ngrams):
  # Get counts
  counts = Counter(ngrams)
  N1 = float(len([k for k in counts if counts[k] == 1]))
  N2 = float(len([k for k in counts if counts[k] == 2]))
  N3 = float(len([k for k in counts if counts[k] == 3]))
  N4 = float(len([k for k in counts if counts[k] == 4]))
  N3p = float(len([k for k in counts if counts[k] >= 3]))

  # Estimate discounting parameters
  Y = N1 / (N1 + 2*N2)
  D1 = 1 - 2*Y*(N2/N1)
  D2 = 2 - 3*Y*(N3/N2)
  D3p = 3 - 4*Y*(N4/N3)

  # TODO(zxie) Check these against someone else's? Couldn't find any.
  #print D1, D2, D3p
  #print N1, N2, N3, N4, N3p

  # FIXME(zxie) Assumes bigrams for now
  # Also compute N1/N2/N3p lookups (context -> n-grams with count 1/2/3+)
  N1_lookup = Counter()
  N2_lookup = Counter()
  N3p_lookup = Counter()
  for bg in counts:
    if counts[bg] == 1:
      N1_lookup[bg[0]] += 1
    elif counts[bg] == 2:
      N2_lookup[bg[0]] += 1
    else:
      N3p_lookup[bg[0]] += 1

  return D1, D2, D3p, N1_lookup, N2_lookup, N3p_lookup

def _read_tokens(filename):
  with open(filename, "r") as f:
    # Note the extra space to ensure </s> is separate token
    tokens = f.read().replace("\n", " </s>").split()
    return tokens

def _compute_freqs(tokens, vocab):
  counter = collections.Counter(tokens)
  ntokens = len(vocab._word_to_id)

  total_count = sum(counter.values())
  frequencies = [counter[vocab.IdToWord(k)]/float(total_count) for k in xrange(ntokens)]

  # Compute number of distinct different histories
  # Currently just bigrams ("bg")
  bg_hist_sets = collections.defaultdict(set)
  for k in xrange(1, len(tokens)):
    bg_hist_sets[tokens[k]].add(tokens[k-1])
  bg_hist_counts = Counter(dict([(k, len(s)) for k, s in bg_hist_sets.iteritems()]))
  total_hists = sum(bg_hist_counts.values())

  hist_freqs = [bg_hist_counts[vocab.IdToWord(k)]/float(total_hists) for k in xrange(ntokens)]

  return frequencies, hist_freqs

def add_blank_token(vocab):
  # Extra token that extends the vocabulary
  next_ind = len(vocab._word_to_id)
  vocab._word_to_id["<_>"] = next_ind
  vocab._id_to_word[next_ind] = "<_>"
  vocab._count += 1

class NgramData(object):

  def __init__(self, data_path, vocab):
    self.vocab= vocab
    tokens = _read_tokens(data_path)

    frequencies, hist_freqs = _compute_freqs(tokens, vocab)
    self.bg_counts = bigram_counts(tokens)
    self.tg_counts = trigram_counts(tokens)

    self.frequencies = frequencies
    self.hist_freqs = hist_freqs
    self.continuations = build_continuations(self.bg_counts)
    bgs = nltk.bigrams(tokens)
    self.D1, self.D2, self.D3p, self.N1_lookup, self.N2_lookup, self.N3p_lookup = estimate_modkn_discounts(bgs)

def corrupt_batch(x, y, lens, flags, ngram_data):
  ngd = ngram_data
  delta = flags.delta
  if delta == 0.0:
    return x, y
  continuations = ngd.continuations
  x_, y_ = np.array(x), np.array(y)

  for row in xrange(x.shape[0]):
    # NOTE skip first <s>
    for col in xrange(1, lens[row]):
      # Compute p
      if flags.noise_scheme == "swap" and flags.swap_scheme in ["ad", "kn", "mkn"]:
        context = list()
        context.append(ngd.id_to_token[x[row, col]])
        # Can also compute D = n1/(n1+n2) as described in Chen & Goodman
        total, distinct = continuations["total"][tuple(context)],\
                continuations["distinct"][tuple(context)]
        if flags.swap_scheme != "mkn":
          p = (delta / total) * distinct
        else:
          p = delta * (ngd.D1 * ngd.N1_lookup[context[0]] +\
                       ngd.D2 * ngd.N2_lookup[context[0]] +\
                       ngd.D3p * ngd.N3p_lookup[context[0]]) / float(total)
      else:
        p = delta
      draw = np.random.binomial(1, p)

      # Determine what to swap in
      if draw:
        if flags.noise_scheme == "drop":
          x_[row, col] = ngd.vocab.WordToId("<_>")
        elif flags.noise_scheme == "swap":
          if flags.swap_scheme == "unigram":
            freqs = ngd.frequencies
          elif flags.swap_scheme == "uniform":
            freqs = np.ones(len(ngd.frequencies), dtype=np.float32) / len(ngd.frequencies)
          elif "kn" in flags.swap_scheme:
            pass
          else:
            assert(False)

          if "kn" not in flags.swap_scheme:
            x_[row, col] = int(np.argmax(np.random.multinomial(1, freqs)))
          else:
            x_[row, col] = int(np.argmax(np.random.multinomial(1, ngd.hist_freqs)))
            y_[row, col] = int(np.argmax(np.random.multinomial(1, ngd.hist_freqs)))
        else:
          raise

  return x_, y_

if __name__ == "__main__":
  import data
  from textsum_cfg import VOCAB_PATH
  from textsum_cfg import TRAIN_SRC_PATH
  from textsum_cfg import TRAIN_TGT_PATH

  # Try things out
  vocab = data.Vocab(VOCAB_PATH, 1000000)
  ngd = NgramData(TRAIN_SRC_PATH, vocab)

  # E.g. "the"
  print("Most frequent token: %s" % (vocab.IdToWord(np.argmax(ngd.frequencies))))
  # E.g. ","
  print("Token with most distinct histories: %s" % (vocab.IdToWord(np.argmax(ngd.hist_freqs))))
  # E.g. "and"
  print("Token with most distinct continuations: %s" % max(ngd.continuations["distinct"].iterkeys(), key=(lambda key: ngd.continuations["distinct"][key])))
  # Should be same as most frequent token
  print("Token with most total continuations: %s" % max(ngd.continuations["total"].iterkeys(), key=(lambda key: ngd.continuations["total"][key])))
