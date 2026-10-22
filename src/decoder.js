// Abbreviation decoder: turns word initials ("wdyw") into full sentences,
// ranked by a language model that is adapted to what the conversation
// partner just said and to the user's own past phrasing.
//
// Every candidate it returns is guaranteed to match the typed initials, which
// is also how LLM suggestions are verified before they are shown.

import { BOS, EOS, Adaptation } from './lm.js';
import { cleanAbbrev, tokenize } from './text.js';

const DEFAULTS = {
  beamWidth: 48,
  perInitial: 120, // most frequent words per initial considered at each step
  maxResults: 5,
  phraseBonus: 4, // log-space bonus for exact matches from the user's phrase bank
};

export class Decoder {
  constructor(lm, opts = {}) {
    this.lm = lm;
    this.opts = { ...DEFAULTS, ...opts };
    this.adapt = new Adaptation(opts.adaptation);
  }

  setContext(turns) {
    this.adapt.setContext(turns || []);
  }

  setPhrases(texts) {
    this.adapt.setPhrases(texts || []);
  }

  wordScore(prevWords, w) {
    const lm = this.lm;
    const n = prevWords.length;
    const u = n > 1 ? lm.id(prevWords[n - 2]) : n === 1 ? lm.id(BOS) : -1;
    const v = n > 0 ? lm.id(prevWords[n - 1]) : lm.id(BOS);
    return this.adapt.score(lm, u, v, w, prevWords);
  }

  // Words that fit a slot constraint: an initial ("w") or a longer spelled
  // prefix ("wa") when the user has typed more letters for that word.
  candidatesFor(prevWords, constraint) {
    const lm = this.lm;
    const initial = constraint[0];
    const pool = new Set();
    const v = prevWords.length ? lm.id(prevWords[prevWords.length - 1]) : lm.id(BOS);
    for (const j of lm.successors(v, initial)) pool.add(lm.vocab[j]);
    const n = constraint.length > 1 ? Infinity : this.opts.perInitial;
    for (const j of lm.topByInitial(initial, n)) pool.add(lm.vocab[j]);
    for (const w of this.adapt.wordsWithInitial(initial)) pool.add(w);
    if (constraint.length > 1) {
      for (const w of pool) if (!w.startsWith(constraint)) pool.delete(w);
      if (!pool.size) pool.add(constraint); // let the user's spelling stand as a word
    }
    return pool;
  }

  sentenceScore(words) {
    let s = 0;
    for (let i = 0; i < words.length; i++) s += this.wordScore(words.slice(0, i), words[i]);
    return s + this.wordScore(words, EOS);
  }

  // `abbrev` is the typed initials ("wdyw") or an array of per-word
  // constraints (["w", "do", "y", "w"]). `prefix` fixes the first words.
  decode(abbrev, { k = this.opts.maxResults, prefix = [] } = {}) {
    const constraints = typeof abbrev === 'string' ? [...cleanAbbrev(abbrev)] : abbrev;
    if (!constraints.length) return [];
    let beam = [{ words: [...prefix], score: prefix.length ? this.prefixScore(prefix) : 0 }];
    for (const c of constraints.slice(prefix.length)) {
      const next = [];
      for (const hyp of beam) {
        for (const w of this.candidatesFor(hyp.words, c)) {
          next.push({ words: [...hyp.words, w], score: hyp.score + this.wordScore(hyp.words, w) });
        }
      }
      next.sort((a, b) => b.score - a.score);
      beam = next.slice(0, this.opts.beamWidth);
      if (!beam.length) return [];
    }
    const results = beam.map((h) => ({ words: h.words, score: h.score + this.wordScore(h.words, EOS), source: 'lm' }));

    // The user's own saved phrases that fit exactly.
    for (const p of this.adapt.phrases) {
      if (fits(p.words, constraints) && prefix.every((w, i) => p.words[i] === w)) {
        results.push({ words: p.words, score: this.sentenceScore(p.words) + this.opts.phraseBonus, source: 'personal' });
      }
    }
    return rankUnique(results, k);
  }

  prefixScore(words) {
    let s = 0;
    for (let i = 0; i < words.length; i++) s += this.wordScore(words.slice(0, i), words[i]);
    return s;
  }

  // Ranked choices for slot `pos`, given the confirmed words before it and
  // the constraints after it (so later initials inform the choice).
  slotOptions(constraints, prefix, k = 5) {
    const pos = prefix.length;
    const ranked = this.decode(constraints, { k: this.opts.beamWidth, prefix });
    const out = [];
    for (const r of ranked) {
      const w = r.words[pos];
      if (!out.includes(w)) out.push(w);
      if (out.length === k) return out;
    }
    const rest = [...this.candidatesFor(prefix, constraints[pos])]
      .filter((w) => !out.includes(w))
      .map((w) => [w, this.wordScore(prefix, w)])
      .sort((a, b) => b[1] - a[1]);
    for (const [w] of rest) {
      out.push(w);
      if (out.length === k) break;
    }
    return out;
  }

  // Merge externally proposed sentences (e.g. from Claude): keep only those
  // that fit the typed constraints and confirmed prefix, score them with the
  // same model, and blend them with the decoder's own results.
  rerank(abbrev, extraTexts, base, { k = this.opts.maxResults, prefix = [], extraBonus = 3 } = {}) {
    const constraints = typeof abbrev === 'string' ? [...cleanAbbrev(abbrev)] : abbrev;
    const merged = [...base];
    extraTexts.forEach((text, rank) => {
      const words = tokenize(text);
      if (!fits(words, constraints) || !prefix.every((w, i) => words[i] === w)) return; // reject violations
      merged.push({ words, score: this.sentenceScore(words) + extraBonus - 0.3 * rank, source: 'llm', text });
    });
    return rankUnique(merged, k);
  }
}

function fits(words, constraints) {
  return words.length === constraints.length && words.every((w, i) => w.startsWith(constraints[i]));
}

function rankUnique(results, k) {
  results.sort((a, b) => b.score - a.score);
  const seen = new Set();
  const out = [];
  for (const r of results) {
    const key = r.words.join(' ');
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(r);
    if (out.length === k) break;
  }
  return out;
}
