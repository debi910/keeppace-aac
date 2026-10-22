// Benchmark core shared by the Node runner (eval/benchmark.mjs) and the
// in-browser runner (bench.html). Simulates a "perfect user", the standard
// assumption in AAC text-entry research: whenever the intended word or
// sentence is on screen, the user picks it.

import { Adaptation } from './lm.js';
import { tokenize, initialsOf } from './text.js';

// Turk Dialogues TSV (set, id, turn1..turn6) -> reply items per split. Each
// turn from the 2nd on is a reply; the replier plays the AAC user.
export function parseTurk(tsv) {
  const splits = {};
  const lines = tsv.split(/\r?\n/);
  if (!lines[0].startsWith('set\tid')) throw new Error('Not a turk-dialogues.txt file');
  for (const line of lines.slice(1)) {
    const cols = line.split('\t');
    if (cols.length < 3) continue;
    const turns = cols.slice(2).map((t) => t.trim()).filter(Boolean);
    const out = (splits[cols[0]] ||= []);
    for (let k = 1; k < turns.length; k++) {
      const context = turns.slice(0, k).map((text, i) => ({ speaker: (k - i) % 2 === 0 ? 'user' : 'partner', text }));
      out.push({ id: `${cols[1]}#${k + 1}`, context, target: turns[k] });
    }
  }
  return splits;
}

// Word-prediction keyboard with 5 slots: type letters until the word shows
// up, then select it (the selection also inserts the space).
export function wordPredCost(lm, words, adapt = null) {
  let cost = 0;
  for (let i = 0; i < words.length; i++) {
    const w = words[i];
    const prev = words.slice(0, i);
    let c = w.length + 1;
    for (let k = 0; k < w.length; k++) {
      if (lm.completions(prev, w.slice(0, k), 5, adapt).includes(w)) {
        c = k + 1;
        break;
      }
    }
    cost += c;
  }
  return cost; // the last word's "space" stands in for the speak key
}

// Correct-and-continue protocol for initials input. After typing initials the
// user sees 5 sentences. If the reply is there: 1 tap. Otherwise they tap the
// first wrong word of the top sentence (1), then pick the right word from the
// slot options (1), typing extra letters of that word first if needed (+1
// each). Earlier words are accepted, the rest is re-decoded, and so on.
export function initialsCost(words, firstRanked, decoder) {
  const n = words.length;
  const key = words.join(' ');
  const constraints = words.map((w) => w[0]);
  let cost = n;
  let prefix = [];
  let ranked = firstRanked;
  let corrections = 0;
  for (;;) {
    if (ranked.slice(0, 5).some((r) => r.words.join(' ') === key)) return { cost: cost + 1, corrections };
    const top = ranked[0] ? ranked[0].words : [];
    let i = prefix.length;
    while (i < n && top[i] === words[i]) i++;
    if (i >= n) return { cost: cost + 1, corrections };
    cost += 1; // tap the wrong word
    corrections++;
    const target = words[i];
    let typed = target[0];
    for (;;) {
      constraints[i] = typed;
      if (decoder.slotOptions(constraints, words.slice(0, i), 5).includes(target)) {
        cost += 1;
        break;
      }
      if (typed === target) {
        cost += 1; // confirm the fully spelled word
        break;
      }
      typed = target.slice(0, typed.length + 1);
      cost += 1;
    }
    prefix = words.slice(0, i + 1);
    if (prefix.length === n) return { cost: cost + 1, corrections };
    ranked = decoder.decode(constraints, { prefix });
  }
}

export const SYSTEMS = {
  letters: 'Spelling, no prediction',
  wordpred: 'Word prediction (typical AAC)',
  'wordpred+ctx': 'Word prediction + conversation',
  'keeppace-noctx': 'KeepPace, no conversation',
  keeppace: 'KeepPace offline',
  'keeppace+llm': 'KeepPace + Claude',
};

// Runs every system on one item and adds to `acc`. `llmCandidates` is the
// list Claude returned for this item, or null to skip the Claude system.
export function scoreItem(acc, item, { lm, plainDecoder, ctxDecoder, llmCandidates = null, llmMs = 0 }) {
  const words = tokenize(item.target);
  if (!words.length) return null;
  const text = words.join(' ');
  acc.n++;
  acc.chars += text.length;
  acc.words += words.length;
  const add = (name, cost, extra = {}) => {
    const t = (acc.systems[name] ||= { keys: 0, hits: 0, corr: 0, ms: 0 });
    t.keys += cost;
    if (extra.corrections === 0) t.hits++;
    t.corr += extra.corrections || 0;
    t.ms += extra.ms || 0;
  };
  const adapt = new Adaptation();
  adapt.setContext(item.context);
  ctxDecoder.setContext(item.context);

  add('letters', text.length + 1);
  add('wordpred', wordPredCost(lm, words));
  add('wordpred+ctx', wordPredCost(lm, words, adapt));

  const abbrev = initialsOf(words);
  const out = { abbrev, text };
  for (const [name, dec] of [['keeppace-noctx', plainDecoder], ['keeppace', ctxDecoder]]) {
    const t0 = performance.now();
    const ranked = dec.decode(abbrev);
    const ms = performance.now() - t0;
    const r = initialsCost(words, ranked, dec);
    add(name, r.cost, { corrections: r.corrections, ms });
    if (name === 'keeppace') out.offlineTop = ranked[0] ? ranked[0].words.join(' ') : '';
  }
  if (llmCandidates) {
    const base = ctxDecoder.decode(abbrev, { k: 20 });
    const ranked = ctxDecoder.rerank(abbrev, llmCandidates, base);
    const r = initialsCost(words, ranked, ctxDecoder);
    add('keeppace+llm', r.cost, { corrections: r.corrections, ms: llmMs });
    out.llmTop = ranked[0] ? ranked[0].words.join(' ') : '';
    out.llmValid = llmCandidates.filter((c) => initialsOf(tokenize(c)) === abbrev).length;
    out.llmTotal = llmCandidates.length;
    acc.llmValid = (acc.llmValid || 0) + out.llmValid;
    acc.llmTotal = (acc.llmTotal || 0) + out.llmTotal;
  }
  return out;
}

export function newAcc() {
  return { n: 0, chars: 0, words: 0, systems: {} };
}

// Time model: every selection (key, suggestion or word) costs one eye-gaze
// dwell; 1.0 s is a typical dwell + saccade for experienced gaze users.
export function summarize(acc, dwell = 1.0) {
  return Object.keys(SYSTEMS)
    .filter((s) => acc.systems[s])
    .map((s) => {
      const t = acc.systems[s];
      const initials = s.startsWith('keeppace');
      return {
        system: s,
        label: SYSTEMS[s],
        keystrokes: t.keys,
        ksr: +(100 * (1 - t.keys / acc.chars)).toFixed(1),
        wpm: +(acc.words / ((t.keys * dwell) / 60)).toFixed(1),
        firstTry: initials ? +((100 * t.hits) / acc.n).toFixed(1) : null,
        fixesPerReply: initials ? +(t.corr / acc.n).toFixed(2) : null,
        msPerReply: initials ? +(t.ms / acc.n).toFixed(0) : null,
      };
    });
}
