# Solutions — Full-Stack AI App

```
02_solutions/
├── backend/            Express API (Node.js) — toy Markov-chain model
│   ├── server.js
│   ├── model/
│   │   ├── markovModel.js
│   │   ├── corpus.txt      (reused from Project 013)
│   │   └── README.md       how to swap in a real trained model
│   ├── package.json
│   └── Dockerfile
├── frontend/           React chat UI (Vite)
│   ├── src/App.jsx
│   ├── src/App.css
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml
└── FINDINGS.md         what was actually run and verified, and what wasn't
```

## Run it yourself

```bash
# terminal 1
cd backend && npm install && node server.js

# terminal 2
cd frontend && npm install && npm run dev
```

Open the printed Vite dev server URL and chat with it — the reply comes
from the toy Markov model described in `backend/model/README.md`, along
with exactly what to change to serve a real trained model instead.

See `FINDINGS.md` for what was actually executed and verified while
authoring this project, and what wasn't (Docker builds — no daemon
running in the authoring sandbox).
