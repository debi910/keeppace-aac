"""Build the n-gram model the app ships with, and the benchmark splits.

Usage:
  python3 tools/build_lm.py --turk data/raw/turk-dialogues.txt [--switchboard DIR] [--out model/lm.json]

Inputs
  --turk         Vertanen's crowdsourced AAC-style dialogues (TSV: set, id, turn1..turn6),
                 from http://keithv.com/data/turk-dialogues.txt. Only the `train` rows feed
                 the model; `dev`/`test` rows are held out for the benchmark.
  --switchboard  Optional NLTK switchboard sample directory, extra conversational text.
Word frequencies from the `wordfreq` package give the model a broad vocabulary.
"""

import argparse
import collections
import json
import math
import os
import re

from wordfreq import top_n_list, word_frequency

BOS, EOS = "<s>", "</s>"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tokenize(text):
    # Mirrors src/text.js tokenize().
    text = re.sub(r"[‘’ʼ`]", "'", text.lower())
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    words = [w.strip("'") for w in text.split()]
    return [w for w in words if w]


def read_turk(path):
    splits = collections.defaultdict(list)
    with open(path, encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\n").split("\t")
        assert header[:2] == ["set", "id"], f"unexpected header {header[:3]}"
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 3:
                continue
            turns = [t.strip() for t in cols[2:] if t.strip()]
            splits[cols[0]].append({"id": cols[1], "turns": turns})
    return splits


def read_switchboard(d):
    out = []
    path = os.path.join(d, "discourse")
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.match(r"\S+ [AB]\.\d+ utt\d+: (.*)", line.strip())
            if m:
                text = re.sub(r"\{[A-Z] ([^}]*)\}", r"\1", m.group(1))
                text = re.sub(r"[\[\]+/#{}-]", " ", text)
                out.append(text)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turk")
    ap.add_argument("--switchboard")
    ap.add_argument("--out", default=os.path.join(HERE, "model", "lm.json"))
    ap.add_argument("--wordfreq-top", type=int, default=25000)
    args = ap.parse_args()

    sentences = []
    if args.turk:
        splits = read_turk(args.turk)
        print({k: len(v) for k, v in splits.items()})
        for d in splits.get("train", []):
            sentences.extend(d["turns"])
    if args.switchboard:
        sentences.extend(read_switchboard(args.switchboard))
    if not sentences:
        raise SystemExit("no training text: pass --turk and/or --switchboard")

    uni, bi, tri = collections.Counter(), collections.Counter(), collections.Counter()
    for s in sentences:
        words = [BOS] + tokenize(s) + [EOS]
        if len(words) == 2:
            continue
        uni.update(words[1:])
        bi.update(zip(words, words[1:]))
        tri.update(zip(words, words[1:], words[2:]))
    total = sum(uni.values())

    wf = [w for w in top_n_list("en", args.wordfreq_top) if re.fullmatch(r"[a-z0-9']+", w)]
    vocab = [BOS, EOS] + sorted(set(w for w in uni if w not in (BOS, EOS)) | set(wf))
    index = {w: i for i, w in enumerate(vocab)}

    prior = []
    for w in vocab:
        pc = uni.get(w, 0) / total
        pw = word_frequency(w, "en") if w not in (BOS, EOS) else 0.0
        p = 0.6 * pc + 0.4 * pw if w != EOS else pc
        prior.append(round(math.log(max(p, 1e-9)), 3))

    bi_out = collections.defaultdict(list)
    for (a, b), c in bi.items():
        bi_out[index[a]].append([index[b], c])
    tri_out = collections.defaultdict(list)
    for (a, b, c3), c in tri.items():
        tri_out[f"{index[a]},{index[b]}"].append([index[c3], c])

    data = {"vocab": vocab, "prior": prior, "bi": bi_out, "tri": tri_out,
            "meta": {"sentences": len(sentences), "tokens": total}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"vocab={len(vocab)} bigram contexts={len(bi_out)} trigram contexts={len(tri_out)} "
          f"size={os.path.getsize(args.out)/1e6:.1f}MB -> {args.out}")


if __name__ == "__main__":
    main()
