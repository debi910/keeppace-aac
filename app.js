import { NgramLM } from './src/lm.js';
import { Decoder } from './src/decoder.js';
import { LLMSuggester, PROVIDERS, listModels } from './src/llm.js';
import { toDisplay, tokenize } from './src/text.js';

const SDK_URL = 'https://cdn.jsdelivr.net/npm/@anthropic-ai/sdk@0.131.0/+esm';
const $ = (id) => document.getElementById(id);

// ---------- persistence (per-browser conveniences only) ----------
const store = {
  get(key, dflt) {
    try {
      const v = localStorage.getItem('keeppace.' + key);
      return v === null ? dflt : JSON.parse(v);
    } catch {
      return dflt;
    }
  },
  set(key, v) {
    try {
      localStorage.setItem('keeppace.' + key, JSON.stringify(v));
    } catch {}
  },
};

const settings = Object.assign(
  { provider: '', keys: {}, model: '', dwell: false, dwellMs: 900, scan: false, scanMs: 1200, voice: '' },
  store.get('settings', {})
);
let myPhrases = store.get('phrases', []); // typed in settings
let history = store.get('history', []); // what the user actually said

// ---------- state ----------
const state = {
  turns: [],
  constraints: [], // per word: initial, or longer spelled prefix
  prefix: [], // confirmed leading words
  editing: null, // slot index being corrected
  ranked: [],
  keys: 0, // selections spent on the current message
  session: { chars: 0, keys: 0, messages: 0 },
  llmStatus: '',
};

let decoder;
let llm = null;
const llmCache = new Map();
let llmTimer = null;
let llmSeq = 0;

// ---------- decoding ----------
function phrases() {
  return [...myPhrases, ...history.slice(-100)];
}

function llmKey() {
  return JSON.stringify([state.constraints, state.prefix, state.turns.slice(-4)]);
}

function recompute() {
  decoder.setContext(state.turns.slice(-4));
  if (!state.constraints.length) {
    state.ranked = [];
  } else {
    const base = decoder.decode(state.constraints, { prefix: state.prefix, k: 20 });
    const extra = llmCache.get(llmKey());
    state.ranked = extra
      ? decoder.rerank(state.constraints, extra, base, { prefix: state.prefix, k: 5 })
      : base.slice(0, 5);
    scheduleLLM();
  }
  render();
}

function scheduleLLM() {
  clearTimeout(llmTimer);
  if (!llm || state.constraints.length < 2 || state.editing !== null) return;
  const key = llmKey();
  if (llmCache.has(key)) return;
  llmTimer = setTimeout(async () => {
    const seq = ++llmSeq;
    state.llmStatus = `Asking ${aiName()}…`;
    renderStatus();
    const t0 = performance.now();
    try {
      const candidates = await llm.suggest({
        abbrev: state.constraints.map((c) => c[0]).join(''),
        context: state.turns.slice(-6),
        prefix: state.prefix,
        phrases: phrases(),
      });
      llmCache.set(key, candidates);
      if (seq === llmSeq) {
        state.llmStatus = `${aiName()} ${((performance.now() - t0) / 1000).toFixed(1)}s`;
        if (key === llmKey()) recompute();
        else renderStatus();
      }
    } catch (e) {
      if (seq === llmSeq) {
        state.llmStatus = `${aiName()} unavailable, using offline model`;
        renderStatus();
      }
      console.warn(e);
    }
  }, 350);
}

async function initLLM() {
  llm = null;
  const provider = settings.provider;
  const apiKey = provider && settings.keys[provider];
  if (!apiKey) return;
  const model = settings.model || PROVIDERS[provider].model;
  try {
    let client = null;
    if (provider === 'claude') {
      const { default: Anthropic } = await import(SDK_URL);
      client = new Anthropic({ apiKey, dangerouslyAllowBrowser: true });
    }
    llm = new LLMSuggester({ provider, apiKey, client, model });
    state.llmStatus = `${PROVIDERS[provider].label.split(' (')[0]} ready`;
  } catch (e) {
    state.llmStatus = 'Could not load the AI provider';
    console.warn(e);
  }
}

function aiName() {
  return llm ? PROVIDERS[llm.provider].label.split(' (')[0] : 'AI';
}

// ---------- actions ----------
function typeKey(ch) {
  state.keys++;
  if (state.editing !== null) {
    state.constraints[state.editing] += ch;
  } else {
    state.constraints.push(ch);
  }
  recompute();
}

function backspace() {
  state.keys++;
  if (state.editing !== null) {
    const c = state.constraints[state.editing];
    if (c.length > 1) state.constraints[state.editing] = c.slice(0, -1);
    else state.editing = null;
  } else {
    state.constraints.pop();
    if (state.prefix.length > state.constraints.length) state.prefix.length = state.constraints.length;
  }
  recompute();
}

function clearMessage() {
  state.constraints = [];
  state.prefix = [];
  state.editing = null;
  state.keys = 0;
  recompute();
}

function startEdit(i) {
  state.keys++;
  const top = state.ranked[0];
  if (!top) return;
  state.prefix = top.words.slice(0, i);
  // Words being re-chosen go back to their typed initial.
  for (let j = i; j < state.constraints.length; j++) state.constraints[j] = state.constraints[j][0];
  state.editing = i;
  recompute();
}

function pickSlot(word) {
  state.keys++;
  const i = state.editing;
  state.prefix = [...state.prefix.slice(0, i), word];
  state.constraints[i] = word;
  state.editing = null;
  if (state.prefix.length === state.constraints.length) {
    say(state.prefix);
    return;
  }
  recompute();
}

function pickSuggestion(r) {
  state.keys++;
  say(r.words);
}

function say(words) {
  const text = toDisplay(words);
  speak(text);
  state.turns.push({ speaker: 'me', text });
  history.push(text);
  history = history.slice(-300);
  store.set('history', history);
  decoder.setPhrases(phrases());
  state.session.chars += text.length;
  state.session.keys += state.keys;
  state.session.messages++;
  state.lastSaving = { chars: text.length, keys: state.keys };
  clearMessage();
}

function addPartner(text) {
  text = text.trim();
  if (!text) return;
  state.turns.push({ speaker: 'partner', text });
  recompute();
}

// ---------- speech ----------
let recognition = null;
let listening = false;

function speak(text) {
  if (!('speechSynthesis' in window)) return;
  const u = new SpeechSynthesisUtterance(text);
  const v = speechSynthesis.getVoices().find((x) => x.name === settings.voice);
  if (v) u.voice = v;
  // Don't transcribe our own voice as the partner.
  if (listening && recognition) {
    recognition.onend = null;
    recognition.stop();
    u.onend = () => listening && startRecognition();
  }
  speechSynthesis.speak(u);
}

function startRecognition() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) return false;
  recognition = new SR();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = 'en-US';
  recognition.onresult = (e) => {
    let interim = '';
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const r = e.results[i];
      if (r.isFinal) addPartner(r[0].transcript);
      else interim += r[0].transcript;
    }
    $('interim').textContent = interim;
  };
  recognition.onend = () => listening && recognition.start();
  recognition.start();
  return true;
}

function toggleListen() {
  if (listening) {
    listening = false;
    if (recognition) {
      recognition.onend = null;
      recognition.stop();
    }
    $('interim').textContent = '';
  } else {
    listening = startRecognition();
    if (!listening) alert('Live listening needs Chrome or Edge. You can type what your partner said instead.');
  }
  $('btn-listen').setAttribute('aria-pressed', String(listening));
  $('btn-listen').textContent = listening ? 'Listening…' : 'Listen';
}

// ---------- rendering ----------
function el(tag, attrs = {}, children = []) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'onclick') n.addEventListener('click', v);
    else if (k === 'text') n.textContent = v;
    else n.setAttribute(k, v);
  }
  for (const c of children) n.append(c);
  return n;
}

function button(label, onclick, cls = '') {
  return el('button', { class: ('target ' + cls).trim(), type: 'button', onclick }, [label]);
}

function render() {
  // conversation
  const turns = $('turns');
  turns.replaceChildren(
    ...state.turns.map((t) =>
      el('li', { class: t.speaker }, [el('small', { text: t.speaker === 'me' ? 'Me' : 'Partner' }), t.text])
    )
  );
  turns.scrollTop = turns.scrollHeight;

  // typed initials, with confirmed words shown in full
  const typed = $('typed');
  if (!state.constraints.length) {
    typed.replaceChildren(el('span', { class: 'placeholder', text: 'Type the first letter of each word' }));
  } else {
    typed.replaceChildren(
      ...state.constraints.map((c, i) => {
        const fixed = i < state.prefix.length;
        const cls = 'slot' + (fixed ? ' fixed' : '') + (state.editing === i ? ' editing' : '');
        return el('span', { class: cls, text: (fixed ? state.prefix[i] : c) + (fixed ? ' ' : '') });
      })
    );
  }

  // sentence suggestions
  $('suggestions').replaceChildren(
    ...state.ranked.map((r, i) => {
      const b = button('', () => pickSuggestion(r));
      b.append(el('span', { class: 'num', text: String(i + 1) }), el('span', { text: toDisplay(r.words) }));
      if (r.source !== 'lm') b.append(el('span', { class: 'badge', text: r.source === 'llm' ? `${aiName()} ✓` : 'yours' }));
      return el('li', {}, [b]);
    })
  );

  // tap-to-fix chips for the top suggestion
  const top = state.ranked[0];
  $('chips-wrap').hidden = !top || state.editing !== null;
  if (top) $('chips').replaceChildren(...top.words.map((w, i) => button(w, () => startEdit(i), 'small')));

  // options for the word being fixed
  const panel = $('slot-panel');
  panel.hidden = state.editing === null;
  if (state.editing !== null) {
    const i = state.editing;
    const opts = decoder.slotOptions(state.constraints, state.prefix.slice(0, i), 6);
    $('slot-hint').textContent = `Word ${i + 1} (“${state.constraints[i]}…”): pick it, or type more letters`;
    $('slot-options').replaceChildren(...opts.map((w) => button(w, () => pickSlot(w), 'small')));
  }
  renderStatus();
}

function renderStatus() {
  $('llm-status').textContent = llm ? state.llmStatus : 'Offline model';
  const s = state.session;
  const parts = [];
  if (state.lastSaving) {
    const { chars, keys } = state.lastSaving;
    parts.push(`Last message: ${keys} selections for ${chars} characters`);
  }
  if (s.messages) parts.push(`Session: ${Math.round(100 * (1 - s.keys / s.chars))}% fewer selections than spelling`);
  $('stats').textContent = parts.join(' · ');
}

function buildKeyboard() {
  const rows = ['1234567890', 'qwertyuiop', 'asdfghjkl', 'zxcvbnm'];
  const kb = $('keyboard');
  rows.forEach((r, ri) => {
    const row = el('div', { class: 'krow', 'data-scan-group': `Keys ${ri + 1}` });
    for (const ch of r) row.append(button(ch, () => typeKey(ch)));
    if (ri === 2) row.append(button('⌫', backspace));
    if (ri === 3) {
      row.append(button('Clear', clearMessage, 'wide'));
      row.append(button('Speak #1', () => state.ranked[0] && pickSuggestion(state.ranked[0]), 'speak'));
    }
    kb.append(row);
  });
}

// ---------- hands-free: dwell ----------
let dwellTarget = null;
let dwellStart = 0;
let dwellDone = false;

function dwellLoop(now) {
  if (settings.dwell && dwellTarget && !dwellDone) {
    const p = Math.min(1, (now - dwellStart) / settings.dwellMs);
    let fill = dwellTarget.querySelector(':scope > .dwell-fill');
    if (!fill) {
      fill = el('span', { class: 'dwell-fill' });
      dwellTarget.append(fill);
    }
    fill.style.transform = `scaleX(${p})`;
    if (p >= 1) {
      dwellDone = true;
      fill.style.transform = 'scaleX(0)';
      dwellTarget.click();
    }
  }
  requestAnimationFrame(dwellLoop);
}

document.addEventListener('pointermove', (e) => {
  if (!settings.dwell) return;
  const t = e.target.closest && e.target.closest('.target');
  if (t !== dwellTarget) {
    const old = dwellTarget && dwellTarget.querySelector(':scope > .dwell-fill');
    if (old) old.style.transform = 'scaleX(0)';
    dwellTarget = t;
    dwellStart = performance.now();
    dwellDone = false; // re-arms only after leaving the target
  }
});

// ---------- hands-free: single-switch row/column scanning ----------
const scan = { level: 0, groups: [], gi: -1, items: [], ii: -1, laps: 0, timer: null };

function scanGroups() {
  return [...document.querySelectorAll('[data-scan-group]')].filter(
    (g) => g.offsetParent !== null && g.querySelector('.target')
  );
}

function clearScanMarks() {
  document.querySelectorAll('.scan-group').forEach((n) => n.classList.remove('scan-group'));
  document.querySelectorAll('.scan-item').forEach((n) => n.classList.remove('scan-item'));
}

function scanStep() {
  clearScanMarks();
  if (scan.level === 0) {
    scan.groups = scanGroups();
    if (!scan.groups.length) return;
    scan.gi = (scan.gi + 1) % scan.groups.length;
    scan.groups[scan.gi].classList.add('scan-group');
  } else {
    scan.ii++;
    if (scan.ii >= scan.items.length) {
      scan.ii = 0;
      if (++scan.laps >= 2) {
        scan.level = 0; // nothing picked: back out to the groups
        return scanStep();
      }
    }
    scan.items[scan.ii].classList.add('scan-item');
  }
}

function scanSelect() {
  if (scan.level === 0) {
    const g = scan.groups[scan.gi];
    if (!g) return;
    scan.items = [...g.querySelectorAll('.target')];
    scan.level = 1;
    scan.ii = -1;
    scan.laps = 0;
  } else {
    const item = scan.items[scan.ii];
    scan.level = 0;
    scan.gi = -1;
    if (item) item.click();
  }
  restartScan();
}

function restartScan() {
  clearInterval(scan.timer);
  clearScanMarks();
  if (!settings.scan) return;
  scanStep();
  scan.timer = setInterval(scanStep, settings.scanMs);
}

// ---------- keyboard / switch input ----------
document.addEventListener('keydown', (e) => {
  if (e.target.closest('input, textarea, select, dialog')) return;
  if (settings.scan && (e.code === 'Space' || e.code === 'Enter')) {
    e.preventDefault();
    scanSelect();
    return;
  }
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (/^[a-z0-9]$/.test(k)) {
    e.preventDefault();
    typeKey(k);
  } else if (k === 'backspace') {
    e.preventDefault();
    backspace();
  } else if (k === 'enter' && state.ranked[0]) {
    e.preventDefault();
    pickSuggestion(state.ranked[0]);
  } else if (k === 'escape') {
    clearMessage();
  }
});

// ---------- settings ----------
function openSettings() {
  $('set-provider').value = settings.provider;
  syncProviderFields(settings.model);
  $('set-dwell').checked = settings.dwell;
  $('set-dwell-ms').value = settings.dwellMs;
  $('set-scan').checked = settings.scan;
  $('set-scan-ms').value = settings.scanMs;
  $('set-phrases').value = myPhrases.join('\n');
  const voices = 'speechSynthesis' in window ? speechSynthesis.getVoices().filter((v) => v.lang.startsWith('en')) : [];
  $('set-voice').replaceChildren(
    el('option', { value: '', text: 'Default voice' }),
    ...voices.map((v) => el('option', { value: v.name, text: `${v.name} (${v.lang})` }))
  );
  $('set-voice').value = settings.voice;
  $('settings').showModal();
}

function setModelOptions(models, selected) {
  $('set-model').replaceChildren(...models.map((m) => el('option', { value: m, text: m })));
  if (selected && models.includes(selected)) $('set-model').value = selected;
}

function syncProviderFields(selected) {
  const p = $('set-provider').value;
  const info = PROVIDERS[p];
  $('set-key').disabled = $('set-model').disabled = $('btn-models').disabled = !p;
  $('set-key').value = (p && settings.keys[p]) || '';
  $('key-link').hidden = !p;
  if (info) {
    $('key-link').href = info.keyUrl;
    $('key-link').textContent = p === 'claude' ? 'Get a key' : 'Get a free key';
  }
  setModelOptions(info ? [...new Set([selected && p === settings.provider ? selected : info.model, info.model])] : [], selected);
}

$('set-provider').addEventListener('change', () => syncProviderFields(null));
$('btn-models').addEventListener('click', async () => {
  const p = $('set-provider').value;
  const btn = $('btn-models');
  btn.textContent = 'Loading…';
  try {
    const models = await listModels(p, $('set-key').value.trim());
    setModelOptions(models, $('set-model').value);
    btn.textContent = `${models.length} models`;
  } catch (e) {
    btn.textContent = 'Check the key';
    console.warn(e);
  }
});

$('settings').addEventListener('close', async () => {
  if ($('settings').returnValue !== 'save') return;
  Object.assign(settings, {
    provider: $('set-provider').value,
    keys: { ...settings.keys, ...($('set-provider').value ? { [$('set-provider').value]: $('set-key').value.trim() } : {}) },
    model: $('set-model').value,
    dwell: $('set-dwell').checked,
    dwellMs: +$('set-dwell-ms').value || 900,
    scan: $('set-scan').checked,
    scanMs: +$('set-scan-ms').value || 1200,
    voice: $('set-voice').value,
  });
  store.set('settings', settings);
  myPhrases = $('set-phrases').value.split('\n').map((s) => s.trim()).filter((s) => tokenize(s).length);
  store.set('phrases', myPhrases);
  decoder.setPhrases(phrases());
  llmCache.clear();
  await initLLM();
  restartScan();
  recompute();
});

// ---------- boot ----------
async function boot() {
  buildKeyboard();
  $('btn-settings').addEventListener('click', openSettings);
  $('btn-listen').addEventListener('click', toggleListen);
  $('partner-form').addEventListener('submit', (e) => {
    e.preventDefault();
    addPartner($('partner-input').value);
    $('partner-input').value = '';
  });
  $('typed').append('Loading language model…');
  const lm = new NgramLM(await (await fetch('model/lm.json')).json());
  decoder = new Decoder(lm);
  decoder.setPhrases(phrases());
  await initLLM();
  requestAnimationFrame(dwellLoop);
  restartScan();
  recompute();
}

boot();
