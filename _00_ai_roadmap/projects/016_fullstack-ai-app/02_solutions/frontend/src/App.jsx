import { useState, useRef, useEffect } from 'react'
import './App.css'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

function App() {
  const [messages, setMessages] = useState([
    { role: 'assistant', text: 'Hi! Ask me anything (the reply comes from a toy Markov-chain model — see the project README for how to swap in a real trained model).' },
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function sendMessage(e) {
    e.preventDefault()
    const text = input.trim()
    if (!text || loading) return

    setMessages((m) => [...m, { role: 'user', text }])
    setInput('')
    setLoading(true)
    setError(null)

    try {
      const res = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.error || `Request failed with status ${res.status}`)
      }
      const data = await res.json()
      setMessages((m) => [...m, { role: 'assistant', text: data.reply }])
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="chat-app">
      <header>
        <h1>Full-Stack AI App</h1>
        <p className="subtitle">React + Express + a toy language model</p>
      </header>

      <main className="chat-window">
        {messages.map((m, i) => (
          <div key={i} className={`bubble ${m.role}`}>
            {m.text}
          </div>
        ))}
        {loading && <div className="bubble assistant loading">thinking&hellip;</div>}
        {error && <div className="bubble error">Error: {error}</div>}
        <div ref={bottomRef} />
      </main>

      <form className="chat-input" onSubmit={sendMessage}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type a message..."
          disabled={loading}
        />
        <button type="submit" disabled={loading || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  )
}

export default App
