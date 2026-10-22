// Benchmark: how many selections (and how much time) does a non-speaking user
// need to say each reply in held-out AAC-style dialogues?
//
//   node eval/benchmark.mjs [--data data/raw/turk-dialogues.txt] [--split test|dev]
//                           [--limit N] [--llm] [--provider gemini] [--model ID] [--json out.json]
//
// --llm adds "KeepPace + AI model". The same benchmark runs in the browser at
// bench.html.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { NgramLM } from '../src/lm.js';
import { Decoder } from '../src/decoder.js';
import { tokenize, initialsOf } from '../src/text.js';
import { parseTurk, scoreItem, newAcc, summarize } from '../src/bench.js';

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const opt = (name, dflt) => {
  const i = args.indexOf('--' + name);
  return i < 0 ? dflt : args[i + 1];
};
const split = opt('split', 'test');
const limit = +opt('limit', 0);
const jsonOut = opt('json', null);
const dataPath = opt('data', path.join(root, 'data', 'raw', 'turk-dialogues.txt'));

const lm = new NgramLM(JSON.parse(fs.readFileSync(path.join(root, 'model', 'lm.json'), 'utf8')));
let items = parseTurk(fs.readFileSync(dataPath, 'utf8'))[split].filter((it) => tokenize(it.target).length);
if (limit) items = items.slice(0, limit);

// --llm [--provider gemini|groq|claude] reads GEMINI_API_KEY, GROQ_API_KEY or
// ANTHROPIC_API_KEY. Suggestions are cached under eval/ so re-runs are free.
let llm = null;
if (args.includes('--llm')) {
  const { LLMSuggester, PROVIDERS, rateLimited } = await import('../src/llm.js');
  const provider = opt('provider', 'gemini');
  let client = null;
  if (provider === 'claude') {
    const { default: Anthropic } = await import('@anthropic-ai/sdk');
    client = new Anthropic();
  }
  const apiKey = { gemini: process.env.GEMINI_API_KEY, groq: process.env.GROQ_API_KEY }[provider];
  const suggester = new LLMSuggester({ provider, apiKey, client, model: opt('model', undefined) });
  const cachePath = path.join(root, 'eval', `llm-cache-${split}-${suggester.model.replace(/\W+/g, '_')}.json`);
  llm = fs.existsSync(cachePath) ? JSON.parse(fs.readFileSync(cachePath, 'utf8')) : {};
  const todo = items.filter((it) => !(it.id in llm));
  let failed = 0;
  await rateLimited(
    todo,
    +opt('rpm', PROVIDERS[provider].rpm),
    async (it) => {
      const t0 = performance.now();
      const candidates = await suggester.suggest({ abbrev: initialsOf(tokenize(it.target)), context: it.context });
      llm[it.id] = { candidates, ms: performance.now() - t0 };
    },
    (done, err) => {
      if (err && ++failed <= 3) console.error('Model call failed:', err.message);
      if (!err && done % 25 === 0) {
        fs.writeFileSync(cachePath, JSON.stringify(llm));
        process.stderr.write(`${suggester.model}: ${done}/${todo.length}\n`);
      }
    }
  );
  fs.writeFileSync(cachePath, JSON.stringify(llm));
  console.error(`${suggester.model}: ${Object.keys(llm).length} cached, ${failed} failed`);
}

const ctx = { lm, plainDecoder: new Decoder(lm), ctxDecoder: new Decoder(lm) };
const acc = newAcc();
const examples = [];
for (const [i, it] of items.entries()) {
  const got = llm && llm[it.id];
  const out = scoreItem(acc, it, { ...ctx, llmCandidates: llm ? (got ? got.candidates : []) : null, llmMs: got ? got.ms : 0 });
  if (out && examples.length < 15) examples.push({ partner: it.context.at(-1).text, ...out });
  if ((i + 1) % 200 === 0) process.stderr.write(`${i + 1}/${items.length}\n`);
}

const rows = summarize(acc);
console.log(`\nKeepPace benchmark: ${acc.n} replies from Turk Dialogues ${split} (${acc.words} words, ${acc.chars} chars)\n`);
console.table(rows.map(({ label, ...r }) => r));
if (acc.llmTotal) console.log(`Model suggestions matching the initials exactly: ${((100 * acc.llmValid) / acc.llmTotal).toFixed(1)}%`);
console.log('\nSamples (partner said | typed | offline top | AI top):');
for (const e of examples) console.log(`  "${e.partner}" | ${e.abbrev} | ${e.offlineTop}${e.llmTop !== undefined ? ' | ' + e.llmTop : ''}  [target: ${e.text}]`);
if (jsonOut) fs.writeFileSync(jsonOut, JSON.stringify({ split, n: acc.n, words: acc.words, chars: acc.chars, rows, examples }, null, 2));
