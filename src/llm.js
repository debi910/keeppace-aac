// Cloud language-model suggestions. The model proposes full replies from the
// conversation, the typed initials and the user's own phrasing; the local
// decoder then rejects any proposal that does not match the initials exactly
// (LLMs often drop or add a word) and blends the rest with its own ranking.
//
// Providers:
//   gemini  Google AI Studio key, free tier (no card needed)
//   groq    Groq key, free tier, very low latency
//   claude  Anthropic API key (paid), via the official SDK

import { cleanAbbrev } from './text.js';

export const PROVIDERS = {
  gemini: { label: 'Google Gemini (free tier)', model: 'gemini-2.5-flash', rpm: 9, keyUrl: 'https://aistudio.google.com/apikey' },
  groq: { label: 'Groq (free tier)', model: 'openai/gpt-oss-120b', rpm: 25, keyUrl: 'https://console.groq.com/keys' },
  claude: { label: 'Claude (paid API)', model: 'claude-opus-5-5', rpm: 50, keyUrl: 'https://console.anthropic.com' },
};
export const DEFAULT_MODEL = PROVIDERS.claude.model;

const SYSTEM = `You help a non-speaking person who communicates by typing only the first letter of each word.
Given the recent conversation and their typed initials, propose the full sentences they most likely mean.
Rules:
- Each sentence must have exactly one word per initial, in order. A contraction like "don't" or "I'm" counts as one word with one initial. Numbers keep their digits (e.g. "7" for seven).
- They are replying to the last thing their partner said, so prefer replies that fit it.
- Sound like them: match the style of their past messages when given.
- Give varied, plausible options, most likely first, no duplicates.
Answer in JSON: {"candidates": ["sentence", ...]}`;

const SCHEMA = {
  type: 'object',
  properties: { candidates: { type: 'array', items: { type: 'string' } } },
  required: ['candidates'],
  additionalProperties: false,
};

export class RateLimitError extends Error {}

export class LLMSuggester {
  // `client` is an Anthropic SDK instance (claude only); others use `apiKey`.
  constructor({ provider = 'claude', apiKey = '', client = null, model, effort = 'low', n = 10 } = {}) {
    this.provider = provider;
    this.apiKey = apiKey;
    this.client = client;
    this.model = model || PROVIDERS[provider].model;
    this.effort = effort;
    this.n = n;
  }

  buildPrompt({ abbrev, context = [], prefix = [], phrases = [] }) {
    const letters = [...cleanAbbrev(abbrev)];
    const convo = context.length
      ? context.map((t) => `${t.speaker === 'partner' ? 'Partner' : 'Me'}: ${t.text}`).join('\n')
      : '(no conversation yet)';
    const slots = letters.map((c, i) => `${i + 1}:${c}`).join(' ');
    let p = `Conversation so far:\n${convo}\n\nMy typed initials: "${letters.join('')}" (${letters.length} words: ${slots})\n`;
    if (prefix.length) p += `My reply definitely starts with: "${prefix.join(' ')}"\n`;
    if (phrases.length) p += `\nThings I often say:\n${phrases.slice(-20).map((x) => `- ${x}`).join('\n')}\n`;
    p += `\nGive ${this.n} candidate sentences.`;
    return p;
  }

  async suggest(args) {
    const prompt = this.buildPrompt(args);
    const text = await this[this.provider](prompt);
    try {
      return JSON.parse(text).candidates.filter((c) => typeof c === 'string');
    } catch {
      return [];
    }
  }

  async claude(prompt) {
    const res = await this.client.beta.messages.create({
      model: this.model,
      max_tokens: 4000,
      betas: ['server-side-fallback-2026-07-01'],
      fallbacks: 'default',
      system: SYSTEM,
      output_config: { effort: this.effort, format: { type: 'json_schema', schema: SCHEMA } },
      messages: [{ role: 'user', content: prompt }],
    });
    if (res.stop_reason === 'refusal') return '{}';
    return res.content.filter((b) => b.type === 'text').map((b) => b.text).join('');
  }

  async gemini(prompt) {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${this.model}:generateContent`;
    const body = await post(url, { 'x-goog-api-key': this.apiKey }, {
      systemInstruction: { parts: [{ text: SYSTEM }] },
      contents: [{ role: 'user', parts: [{ text: prompt }] }],
      generationConfig: {
        responseMimeType: 'application/json',
        responseSchema: { type: 'OBJECT', properties: { candidates: { type: 'ARRAY', items: { type: 'STRING' } } }, required: ['candidates'] },
      },
    });
    const parts = body.candidates?.[0]?.content?.parts || [];
    return parts.filter((p) => !p.thought).map((p) => p.text || '').join('');
  }

  async groq(prompt) {
    const body = await post('https://api.groq.com/openai/v1/chat/completions', { authorization: `Bearer ${this.apiKey}` }, {
      model: this.model,
      response_format: { type: 'json_object' },
      messages: [
        { role: 'system', content: SYSTEM },
        { role: 'user', content: prompt },
      ],
    });
    return body.choices?.[0]?.message?.content || '';
  }
}

async function post(url, headers, payload) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'content-type': 'application/json', ...headers },
    body: JSON.stringify(payload),
  });
  if (res.status === 429) throw new RateLimitError('Rate limited');
  if (!res.ok) throw new Error(`${res.status}: ${(await res.text()).slice(0, 300)}`);
  return res.json();
}

// Models this key can use, for the settings dropdown.
export async function listModels(provider, apiKey) {
  if (provider === 'gemini') {
    const res = await fetch('https://generativelanguage.googleapis.com/v1beta/models?pageSize=200', { headers: { 'x-goog-api-key': apiKey } });
    if (!res.ok) throw new Error(`${res.status}: ${(await res.text()).slice(0, 200)}`);
    const { models = [] } = await res.json();
    return models
      .filter((m) => (m.supportedGenerationMethods || []).includes('generateContent') && /gemini|gemma/.test(m.name))
      .filter((m) => !/image|tts|audio|live|embedding|vision/.test(m.name))
      .map((m) => m.name.replace(/^models\//, ''));
  }
  if (provider === 'groq') {
    const res = await fetch('https://api.groq.com/openai/v1/models', { headers: { authorization: `Bearer ${apiKey}` } });
    if (!res.ok) throw new Error(`${res.status}: ${(await res.text()).slice(0, 200)}`);
    const { data = [] } = await res.json();
    return data.filter((m) => !/whisper|tts|guard|playai/.test(m.id)).map((m) => m.id);
  }
  return ['claude-opus-5-5', 'claude-sonnet-5-5', 'claude-haiku-5-5'];
}

// Runs `fn` over items with a requests-per-minute cap and retries on 429.
export async function rateLimited(items, rpm, fn, onProgress = () => {}) {
  const gap = 60000 / rpm;
  let nextAt = 0;
  let done = 0;
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  const slot = async () => {
    const now = Date.now();
    const at = Math.max(now, nextAt);
    nextAt = at + gap;
    if (at > now) await wait(at - now);
  };
  let idx = 0;
  const worker = async () => {
    while (idx < items.length) {
      const it = items[idx++];
      for (let attempt = 0; attempt < 4; attempt++) {
        await slot();
        try {
          await fn(it);
          break;
        } catch (e) {
          if (e instanceof RateLimitError || /429|rate/i.test(e.message)) await wait(15000 * (attempt + 1));
          else {
            onProgress(done, e);
            break;
          }
        }
      }
      onProgress(++done, null);
    }
  };
  await Promise.all(Array.from({ length: Math.min(8, items.length) }, worker));
}
