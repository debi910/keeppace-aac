// Compact trigram language model with stupid backoff, plus two adaptive
// components that a static model or a chatbot prompt does not have:
//   - a conversation cache built from what the partner just said
//   - a personal model built from the user's own past messages
// Scores are log-probabilities (not perfectly normalized under backoff, which
// is fine for ranking candidates).

import { STOPWORDS, PRONOUN_FLIP, tokenize } from './text.js';

const BACKOFF = 0.4;
const OOV_LOGP = Math.log(1e-8);
export const BOS = '<s>';
export const EOS = '</s>';

export class NgramLM {
  constructor(data) {
    this.vocab = data.vocab;
    this.index = new Map(this.vocab.map((w, i) => [w, i]));
    this.prior = Float64Array.from(data.prior); // log P_unigram
    this.bi = new Map();
    this.biTotal = new Map();
    for (const [k, list] of Object.entries(data.bi)) {
      const m = new Map();
      let t = 0;
      for (const [j, c] of list) {
        m.set(j, c);
        t += c;
      }
      this.bi.set(+k, m);
      this.biTotal.set(+k, t);
    }
    this.tri = new Map();
    this.triTotal = new Map();
    for (const [k, list] of Object.entries(data.tri)) {
      const m = new Map();
      let t = 0;
      for (const [j, c] of list) {
        m.set(j, c);
        t += c;
      }
      this.tri.set(k, m);
      this.triTotal.set(k, t);
    }
    // Words grouped by first character, most frequent first.
    this.byInitial = new Map();
    const order = [...this.vocab.keys()].sort((a, b) => this.prior[b] - this.prior[a]);
    for (const i of order) {
      const w = this.vocab[i];
      if (w === BOS || w === EOS) continue;
      const c = w[0];
      if (!this.byInitial.has(c)) this.byInitial.set(c, []);
      this.byInitial.get(c).push(i);
    }
    this.succCache = new Map();
  }

  id(word) {
    const i = this.index.get(word);
    return i === undefined ? -1 : i;
  }

  // log S(w | u v) with stupid backoff; ids may be -1 for unknown words.
  logp(u, v, w) {
    if (w < 0) return OOV_LOGP;
    let mult = 0;
    if (u >= 0 && v >= 0) {
      const key = u + ',' + v;
      const m = this.tri.get(key);
      const c = m && m.get(w);
      if (c) return Math.log(c / this.triTotal.get(key));
      if (m) mult += Math.log(BACKOFF);
    }
    if (v >= 0) {
      const m = this.bi.get(v);
      const c = m && m.get(w);
      if (c) return mult + Math.log(c / this.biTotal.get(v));
      if (m) mult += Math.log(BACKOFF);
    }
    return mult + this.prior[w];
  }

  // Successor word ids of `v` that start with `initial`, from bigram data.
  successors(v, initial) {
    if (v < 0) return [];
    let byInit = this.succCache.get(v);
    if (!byInit) {
      byInit = new Map();
      const m = this.bi.get(v);
      if (m) {
        for (const j of m.keys()) {
          const w = this.vocab[j];
          if (w === EOS) continue;
          const c = w[0];
          if (!byInit.has(c)) byInit.set(c, []);
          byInit.get(c).push(j);
        }
      }
      this.succCache.set(v, byInit);
    }
    return byInit.get(initial) || [];
  }

  topByInitial(initial, n) {
    return (this.byInitial.get(initial) || []).slice(0, n);
  }

  // Word-completion candidates for a typed prefix (used by the baseline
  // word-prediction keyboard and by the app's spell mode).
  completions(prevWords, prefix, k = 5, adapt = null) {
    const ids = prevWords.map((w) => this.id(w));
    const u = ids.length > 1 ? ids[ids.length - 2] : this.id(BOS);
    const v = ids.length > 0 ? ids[ids.length - 1] : this.id(BOS);
    const pool = new Set();
    if (prefix) {
      for (const j of this.successors(v, prefix[0])) pool.add(this.vocab[j]);
      for (const j of this.topByInitial(prefix[0], 400)) pool.add(this.vocab[j]);
    } else {
      const m = this.bi.get(v);
      if (m) for (const j of m.keys()) pool.add(this.vocab[j]);
      for (let c of 'abcdefghijklmnopqrstuvwxyz') for (const j of this.topByInitial(c, 20)) pool.add(this.vocab[j]);
    }
    if (adapt) for (const w of adapt.words()) pool.add(w);
    pool.delete(EOS);
    const scored = [];
    for (const w of pool) {
      if (prefix && !w.startsWith(prefix)) continue;
      if (w.length <= prefix.length) continue;
      const s = adapt ? adapt.score(this, u, v, w, prevWords) : this.logp(u, v, this.id(w));
      scored.push([w, s]);
    }
    scored.sort((a, b) => b[1] - a[1]);
    return scored.slice(0, k).map((x) => x[0]);
  }
}

// Conversation cache + personal model, mixed into the base LM.
export class Adaptation {
  constructor({ cacheWeight = 0.12, personalWeight = 0.15 } = {}) {
    this.cacheWeight = cacheWeight;
    this.personalWeight = personalWeight;
    this.cache = new Map(); // word -> weight
    this.cacheTotal = 0;
    this.pUni = new Map();
    this.pBi = new Map();
    this.pTotal = 0;
    this.phrases = []; // [{text, words}]
  }

  // Partner turns weigh more than older history; content words and mirrored
  // pronouns are what a reply is likely to reuse.
  setContext(turns) {
    this.cache.clear();
    this.cacheTotal = 0;
    turns.forEach((turn, idx) => {
      const recency = turns.length - idx === 1 ? 1 : 0.35;
      const speakerW = turn.speaker === 'partner' ? 1 : 0.5;
      for (const w of tokenize(turn.text)) {
        const base = STOPWORDS.has(w) ? 0.3 : 2;
        this.addCache(w, base * recency * speakerW);
        for (const f of PRONOUN_FLIP[w] || []) this.addCache(f, 0.8 * recency * speakerW);
      }
    });
  }

  addCache(w, x) {
    this.cache.set(w, (this.cache.get(w) || 0) + x);
    this.cacheTotal += x;
  }

  setPhrases(texts) {
    this.pUni.clear();
    this.pBi.clear();
    this.pTotal = 0;
    this.phrases = [];
    for (const text of texts) {
      const words = tokenize(text);
      if (!words.length) continue;
      this.phrases.push({ text, words });
      let prev = BOS;
      for (const w of [...words, EOS]) {
        this.pUni.set(w, (this.pUni.get(w) || 0) + 1);
        const k = prev + ' ' + w;
        this.pBi.set(k, (this.pBi.get(k) || 0) + 1);
        this.pTotal++;
        prev = w;
      }
    }
  }

  words() {
    return new Set([...this.cache.keys(), ...this.pUni.keys()]);
  }

  wordsWithInitial(c) {
    const out = [];
    for (const w of this.words()) if (w[0] === c) out.push(w);
    return out;
  }

  // log P(w | history) mixing base LM, conversation cache and personal model.
  score(lm, u, v, w, prevWords) {
    const base = Math.exp(lm.logp(u, v, lm.id(w)));
    let p = base;
    if (this.cacheTotal > 0) {
      const pc = (this.cache.get(w) || 0) / this.cacheTotal;
      p = (1 - this.cacheWeight) * p + this.cacheWeight * pc;
    }
    if (this.pTotal > 0) {
      const prev = prevWords.length ? prevWords[prevWords.length - 1] : BOS;
      const prevCount = this.pUni.get(prev) || 0;
      const bi = this.pBi.get(prev + ' ' + w) || 0;
      const pp = prevCount ? 0.7 * (bi / prevCount) + 0.3 * ((this.pUni.get(w) || 0) / this.pTotal) : (this.pUni.get(w) || 0) / this.pTotal;
      p = (1 - this.personalWeight) * p + this.personalWeight * pp;
    }
    return Math.log(Math.max(p, 1e-12));
  }
}
