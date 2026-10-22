// Claude-backed suggestions. Claude proposes full replies from the
// conversation, the typed initials and the user's own phrasing; the local
// decoder then rejects any proposal that does not match the initials exactly
// (LLMs often drop or add a word) and blends the rest with its own ranking.
// Works with any client exposing the Anthropic SDK's `beta.messages.create`.

import { cleanAbbrev } from './text.js';

export const DEFAULT_MODEL = 'claude-opus-5-5';

const SYSTEM = `You help a non-speaking person who communicates by typing only the first letter of each word.
Given the recent conversation and their typed initials, propose the full sentences they most likely mean.
Rules:
- Each sentence must have exactly one word per initial, in order. A contraction like "don't" or "I'm" counts as one word with one initial. Numbers keep their digits (e.g. "7" for seven).
- They are replying to the last thing their partner said, so prefer replies that fit it.
- Sound like them: match the style of their past messages when given.
- Give varied, plausible options, most likely first, no duplicates.`;

const SCHEMA = {
  type: 'object',
  properties: { candidates: { type: 'array', items: { type: 'string' } } },
  required: ['candidates'],
  additionalProperties: false,
};

export class LLMSuggester {
  constructor(client, { model = DEFAULT_MODEL, effort = 'low', n = 10 } = {}) {
    this.client = client;
    this.model = model;
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
    const res = await this.client.beta.messages.create({
      model: this.model,
      max_tokens: 4000,
      betas: ['server-side-fallback-2026-07-01'],
      fallbacks: 'default',
      system: SYSTEM,
      output_config: { effort: this.effort, format: { type: 'json_schema', schema: SCHEMA } },
      messages: [{ role: 'user', content: this.buildPrompt(args) }],
    });
    if (res.stop_reason === 'refusal') return [];
    const text = res.content.filter((b) => b.type === 'text').map((b) => b.text).join('');
    try {
      return JSON.parse(text).candidates.filter((c) => typeof c === 'string');
    } catch {
      return [];
    }
  }
}
