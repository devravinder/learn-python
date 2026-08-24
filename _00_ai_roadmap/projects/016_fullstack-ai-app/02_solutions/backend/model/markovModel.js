// Order-2 word-level Markov chain, trained in-memory at server startup.
// Deliberately simple and fully inspectable — see model/README.md for
// how this seam would be replaced by a real trained model in production.
const fs = require("fs");
const path = require("path");

class MarkovModel {
  constructor(order = 2) {
    this.order = order;
    this.chain = new Map(); // key: "word1|word2" -> array of possible next words
    this.starts = []; // valid starting n-grams, for when no seed overlaps the corpus
  }

  _key(words) {
    return words.join("|");
  }

  train(text) {
    const words = text
      .replace(/\s+/g, " ")
      .trim()
      .split(" ")
      .filter(Boolean);

    if (words.length <= this.order) {
      throw new Error("Corpus too small to train a Markov chain of this order");
    }

    for (let i = 0; i <= words.length - this.order - 1; i++) {
      const gram = words.slice(i, i + this.order);
      const next = words[i + this.order];
      const key = this._key(gram);
      if (!this.chain.has(key)) {
        this.chain.set(key, []);
        this.starts.push(gram);
      }
      this.chain.get(key).push(next);
    }
  }

  _pickSeed(promptWords) {
    // Try to find a seed n-gram overlapping the user's prompt (so the
    // reply feels at least loosely related to what they typed); fall
    // back to a random n-gram from the corpus otherwise.
    const lowerPrompt = promptWords.map((w) => w.toLowerCase());
    for (const gram of this.starts) {
      if (gram.some((w) => lowerPrompt.includes(w.toLowerCase()))) {
        return gram;
      }
    }
    return this.starts[Math.floor(Math.random() * this.starts.length)];
  }

  generate(prompt, maxWords = 40) {
    const promptWords = prompt.trim().split(/\s+/).filter(Boolean);
    let current = this._pickSeed(promptWords);
    const output = [...current];

    for (let i = 0; i < maxWords; i++) {
      const key = this._key(current);
      const candidates = this.chain.get(key);
      if (!candidates || candidates.length === 0) break;
      const next = candidates[Math.floor(Math.random() * candidates.length)];
      output.push(next);
      current = [...current.slice(1), next];
    }

    return output.join(" ");
  }
}

function loadModel(corpusPath = path.join(__dirname, "corpus.txt")) {
  const text = fs.readFileSync(corpusPath, "utf-8");
  const model = new MarkovModel(2);
  model.train(text);
  return model;
}

module.exports = { MarkovModel, loadModel };
