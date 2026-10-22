// Shared text normalization so the app, the decoder and the benchmark agree
// on what a "word" is and what counts as an exact match.

const APOSTROPHES = /[‘’ʼ`]/g;

export function tokenize(text) {
  return text
    .toLowerCase()
    .replace(APOSTROPHES, "'")
    .replace(/[^a-z0-9' ]+/g, ' ')
    .split(/\s+/)
    .map((w) => w.replace(/^'+|'+$/g, ''))
    .filter(Boolean);
}

export function normalize(text) {
  return tokenize(text).join(' ');
}

// "what do you want" -> "wdyw". Contractions keep a single initial ("i'm" -> "i").
export function initialsOf(words) {
  return words.map((w) => w[0]).join('');
}

// Abbreviation as typed by the user: letters/digits only, lowercased.
export function cleanAbbrev(abbrev) {
  return abbrev.toLowerCase().replace(/[^a-z0-9]/g, '');
}

export function matchesInitials(text, abbrev) {
  return initialsOf(tokenize(text)) === cleanAbbrev(abbrev);
}

// Sentence-case a decoded word sequence for display / speech.
export function toDisplay(words) {
  if (!words.length) return '';
  const out = words.map((w) => (w === 'i' || w.startsWith("i'") ? 'I' + w.slice(1) : w));
  out[0] = out[0][0].toUpperCase() + out[0].slice(1);
  return out.join(' ');
}

export const STOPWORDS = new Set(
  ("a an the and or but if of to in on at for with from by as is am are was were be been being do does did " +
    "have has had i you he she it we they me him her us them my your his its our their this that these those " +
    "what which who whom whose when where why how there here not no yes so just very can could will would shall " +
    "should may might must up down out about into over than then too also all any some").split(' ')
);

// When a partner asks "are you hungry?", the reply tends to use the mirrored
// pronoun ("i am hungry"). Used to build the conversation cache.
export const PRONOUN_FLIP = {
  you: ['i', 'me'],
  your: ['my'],
  yours: ['mine'],
  yourself: ['myself'],
  "you're": ["i'm"],
  "you've": ["i've"],
  "you'll": ["i'll"],
  "you'd": ["i'd"],
  i: ['you'],
  me: ['you'],
  my: ['your'],
  mine: ['yours'],
  "i'm": ["you're"],
  are: ['am'],
  am: ['are'],
  we: ['we'],
};
