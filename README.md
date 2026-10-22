# KeepPace

**A keyboard for people who can't speak, built so they can keep up with a live conversation.**

People who use AAC (augmentative and alternative communication) and spell letter by letter, for example with eye gaze, a head mouse or a single switch, often manage only 5 to 10 words a minute. Speech runs at about 150. By the time a reply is ready, the conversation has moved on.

With KeepPace the user types only the **first letter of each word**. KeepPace listens to what the other person just said and turns the initials into full sentences:

> Partner: *"Can you take me somewhere?"*
> User types `wdywtg` → **"Where do you want to go?"** → spoken aloud

## How it works

1. **Listening.** The browser transcribes the partner's speech live (Web Speech API in Chrome or Edge), or the partner's words can be typed in.
2. **Offline decoder.** A trigram language model, trained on crowdsourced AAC-style dialogues, runs a beam search over every sentence that fits the initials. Two things adapt it to the moment:
   - a *conversation cache* that boosts the words the partner just used and their mirrored pronouns ("are you…" → "I am…"),
   - a *personal model* built from the user's saved phrases and everything they have said before, so suggestions sound like them.
3. **Claude, verified.** With an API key, Claude proposes replies from the conversation, the initials and the user's phrasing. Language models often drop or add a word, so KeepPace **rejects any suggestion that does not match the typed initials exactly** and re-ranks the rest together with its own candidates. The user never sees a sentence they did not "type".
4. **Tap to fix.** If no sentence is right, the user taps the first wrong word and picks from options ranked using the *later* initials too, typing more letters only if needed. The rest of the sentence is re-decoded around the fix.
5. **Hands-free.** Everything works with dwell selection (eye tracker, head mouse) or single-switch row/column scanning.

The app is static HTML and JavaScript with no build step and no server. Without a key it runs fully offline in the browser.

## How it compares

| | Speeds up | Uses the partner's words | Sounds like the user | Measured |
|---|---|---|---|---|
| Word prediction (standard in AAC apps) | each word | no | partly | yes, widely |
| Lingraphica *Conversations* | whole replies, written for the user | yes | not yet (planned) | 13-person pilot |
| PRC-Saltillo AI tools | grammar fix, translation, images | no | no | no |
| Google *SpeakFaster* (research) | initials → sentences | yes | no | yes, with a fine-tuned 64B model, not released |
| **KeepPace** | initials → sentences, word-level fixes | yes | yes | yes, open benchmark below |

KeepPace keeps the user in control of every word (unlike reply generators), runs in any browser with or without a cloud model, and ships the benchmark so anyone can check the claims.

## Benchmark

`bench.html` (or `npm run bench`) replays held-out replies from Keith Vertanen's crowdsourced Turk Dialogues and counts how many selections each method needs. It simulates a *perfect user* who always picks the right suggestion when it is on screen, which is the standard assumption in AAC text-entry research. Every reply must be completed, so failures are charged in full: a wrong guess costs taps to fix it.

Words per minute assume one selection per second, a typical pace for experienced eye-gaze users.

Test split, 1,400 replies (11,294 words):

| Method | Selections | Saved vs spelling | Words/min |
|---|---|---|---|
| Spelling, no prediction | 55,062 | -2.6% | 12.3 |
| Word prediction (typical AAC) | 26,100 | 51.4% | 26.0 |
| Word prediction + KeepPace's conversation adaptation | 25,389 | 52.7% | 26.7 |
| KeepPace offline, no conversation | 30,840 | 42.5% | 22.0 |
| KeepPace offline | 29,830 | 44.4% | 22.7 |
| KeepPace + AI model | *pending* | | |

What this shows so far:
- Listening to the partner helps every method: it saves selections for word prediction and for initials.
- The small offline model alone is **not** better than a mature word predictor. Expanding initials into a whole sentence needs a strong language model, which is why the AI layer exists. Published research with a fine-tuned 64B model reached about 74% top-5 accuracy on this data with one turn of context (Cai et al., 2022).

## Run it

```bash
python3 -m http.server 8000      # then open http://localhost:8000
```

Optional: open **Settings** and paste a Claude API key from [console.anthropic.com](https://console.anthropic.com). The key is stored only in your browser and sent only to Anthropic.

Benchmark from the command line:

```bash
# get http://keithv.com/data/turk-dialogues.txt into data/raw/
npm run bench                                  # offline systems, 1,400 test replies
npm install && ANTHROPIC_API_KEY=... node eval/benchmark.mjs --llm --limit 300
```

Rebuild the language model: `pip install wordfreq && npm run build-model`.

## Layout

| Path | What it is |
|---|---|
| `index.html`, `app.js`, `styles.css` | The keyboard |
| `src/decoder.js` | Beam-search initials decoder, slot options, verification of outside suggestions |
| `src/lm.js` | Trigram model, conversation cache, personal model |
| `src/llm.js` | AI suggestions (Gemini, Groq or Claude) |
| `src/bench.js`, `bench.html`, `eval/benchmark.mjs` | Benchmark core, browser runner, Node runner |
| `tools/build_lm.py` | Builds `model/lm.json` from the training dialogues and `wordfreq` |

## Limitations

- Tested with simulated users only. No AAC user has tried it yet, and real users make typos and do not always spot the right suggestion.
- The offline model is trained on about 5,000 sentences, so on its own it is weaker than a mature word predictor. The large gains need Claude.
- Live transcription uses the browser's speech service, which sends audio to the browser vendor. Claude suggestions send the recent conversation to Anthropic. Offline mode sends nothing.
- English only.

## Credits and disclosure

- Dialogue data: K. Vertanen, *Towards improving predictive AAC using crowdsourced dialogues and partner context*, ASSETS 2017. The data file is not included in this repository.
- Word frequencies: [wordfreq](https://github.com/rspeer/wordfreq).
- Built during the Taskforge hackathon with help from Claude (Anthropic) for research, code and writing.
