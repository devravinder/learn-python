const express = require("express");
const cors = require("cors");
const { loadModel } = require("./model/markovModel");

const app = express();
app.use(cors());
app.use(express.json());

const state = { model: null, loaded: false };

try {
  state.model = loadModel();
  state.loaded = true;
  console.log("Model loaded: Markov chain trained on bundled corpus");
} catch (err) {
  console.error("Model failed to load:", err.message);
  state.loaded = false;
}

// Mirrors Lesson 078's FastAPI /health pattern, in Express.
app.get("/api/health", (req, res) => {
  res.json({ status: state.loaded ? "ok" : "degraded", model_loaded: state.loaded });
});

app.post("/api/chat", (req, res) => {
  const { message } = req.body || {};
  if (!message || typeof message !== "string" || !message.trim()) {
    return res.status(422).json({ error: "message must be a non-empty string" });
  }
  if (!state.loaded) {
    return res.status(503).json({ error: "model not loaded" });
  }
  const reply = state.model.generate(message);
  res.json({ reply });
});

const PORT = process.env.PORT || 8000;
if (require.main === module) {
  app.listen(PORT, () => console.log(`Server listening on port ${PORT}`));
}

module.exports = app;
