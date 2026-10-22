import { NgramLM } from './src/lm.js';
import { Decoder } from './src/decoder.js';
import { LLMSuggester, PROVIDERS, rateLimited } from './src/llm.js';
import { tokenize, initialsOf } from './src/text.js';
import { parseTurk, scoreItem, newAcc, summarize } from './src/bench.js';

const SDK_URL = 'https://cdn.jsdelivr.net/npm/@anthropic-ai/sdk@0.131.0/+esm';
const $ = (id) => document.getElementById(id);

let lm;
let splits = null;
let lastResult = null;

function settings() {
  try {
    return JSON.parse(localStorage.getItem('keeppace.settings')) || {};
  } catch {
    return {};
  }
}

$('file').addEventListener('change', async () => {
  const f = $('file').files[0];
  if (!f) return;
  try {
    splits = parseTurk(await f.text());
    $('status').textContent = Object.entries(splits).map(([k, v]) => `${k}: ${v.length} replies`).join(', ');
    $('run').disabled = false;
  } catch (e) {
    $('status').textContent = e.message;
  }
});

$('run').addEventListener('click', run);
$('download').addEventListener('click', () => {
  const blob = new Blob([JSON.stringify(lastResult, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `keeppace-bench-${lastResult.split}.json`;
  a.click();
});

async function run() {
  $('run').disabled = true;
  const split = $('split').value;
  const items = splits[split].filter((it) => tokenize(it.target).length).slice(0, +$('limit').value || undefined);
  const useLLM = $('use-llm').checked;
  const s = settings();
  const progress = $('progress');
  progress.hidden = false;

  let llmResults = null;
  let modelName = null;
  if (useLLM) {
    const provider = s.provider;
    const apiKey = provider && (s.keys || {})[provider];
    if (!apiKey) {
      $('status').textContent = 'Choose an AI provider and add its key in Settings on the keyboard page first.';
      $('run').disabled = false;
      progress.hidden = true;
      return;
    }
    let client = null;
    if (provider === 'claude') {
      const { default: Anthropic } = await import(SDK_URL);
      client = new Anthropic({ apiKey, dangerouslyAllowBrowser: true });
    }
    const suggester = new LLMSuggester({ provider, apiKey, client, model: s.model || PROVIDERS[provider].model });
    modelName = suggester.model;
    llmResults = {};
    let failed = 0;
    progress.max = items.length;
    const rpm = PROVIDERS[provider].rpm;
    await rateLimited(
      items,
      rpm,
      async (it) => {
        const t0 = performance.now();
        const candidates = await suggester.suggest({ abbrev: initialsOf(tokenize(it.target)), context: it.context });
        llmResults[it.id] = { candidates, ms: performance.now() - t0 };
      },
      (done, err) => {
        if (err) {
          failed++;
          console.warn(err);
          return;
        }
        progress.value = done;
        const mins = Math.ceil((items.length - done) / rpm);
        $('status').textContent = `Asking ${modelName}: ${done}/${items.length}${failed ? `, ${failed} failed` : ''} (free-tier pace, about ${mins} min left)`;
      }
    );
  }

  const ctx = { lm, plainDecoder: new Decoder(lm), ctxDecoder: new Decoder(lm) };
  const acc = newAcc();
  const samples = [];
  progress.max = items.length;
  for (const [i, it] of items.entries()) {
    const got = llmResults && llmResults[it.id];
    const out = scoreItem(acc, it, { ...ctx, llmCandidates: llmResults ? (got ? got.candidates : []) : null, llmMs: got ? got.ms : 0 });
    if (out && samples.length < 20) samples.push({ partner: it.context.at(-1).text, ...out });
    if (i % 10 === 0) {
      progress.value = i;
      $('status').textContent = `Scoring ${i}/${items.length}`;
      await new Promise((r) => setTimeout(r)); // keep the page responsive
    }
  }
  progress.hidden = true;
  const rows = summarize(acc);
  lastResult = { split, n: acc.n, words: acc.words, chars: acc.chars, model: modelName, rows, samples, llmValid: acc.llmValid, llmTotal: acc.llmTotal };
  $('status').textContent = `${acc.n} replies, ${acc.words} words.`;
  render(lastResult);
  $('run').disabled = false;
  $('download').disabled = false;
}

function render(res) {
  const max = Math.max(...res.rows.map((r) => r.wpm));
  $('bars').replaceChildren(
    ...res.rows.map((r) => {
      const row = document.createElement('div');
      row.className = 'barrow';
      const bar = document.createElement('div');
      bar.className = 'meter';
      const fill = document.createElement('span');
      fill.style.width = `${(100 * r.wpm) / max}%`;
      if (r.system.startsWith('keeppace')) fill.className = 'ours';
      bar.append(fill);
      row.append(Object.assign(document.createElement('span'), { textContent: r.label }), bar, Object.assign(document.createElement('strong'), { textContent: `${r.wpm} wpm` }));
      return row;
    })
  );
  const head = ['Method', 'Selections', 'Saved vs spelling', 'WPM', 'Right first try', 'Fixes / reply', 'ms / reply'];
  const fmt = (v, suffix = '') => (v === null || v === undefined ? '' : v + suffix);
  $('table').innerHTML = '';
  const thead = $('table').createTHead().insertRow();
  head.forEach((h) => (thead.insertCell().outerHTML = `<th>${h}</th>`));
  const tbody = $('table').createTBody();
  for (const r of res.rows) {
    const tr = tbody.insertRow();
    [r.label, r.keystrokes, fmt(r.ksr, '%'), r.wpm, fmt(r.firstTry, '%'), fmt(r.fixesPerReply), fmt(r.msPerReply)].forEach((v) => (tr.insertCell().textContent = v));
  }
  $('llm-valid').textContent = res.llmTotal
    ? `${((100 * res.llmValid) / res.llmTotal).toFixed(1)}% of ${res.model}'s raw suggestions matched the typed initials; the rest were filtered out before display.`
    : '';
  $('samples').replaceChildren(
    ...res.samples.map((s) => {
      const li = document.createElement('li');
      li.textContent = `Partner: “${s.partner}” → typed ${s.abbrev} → ${s.llmTop ?? s.offlineTop}${(s.llmTop ?? s.offlineTop) === s.text ? ' ✓' : `  (meant: ${s.text})`}`;
      return li;
    })
  );
}

lm = new NgramLM(await (await fetch('model/lm.json')).json());
$('status').textContent = 'Choose the dialogue file to begin.';
