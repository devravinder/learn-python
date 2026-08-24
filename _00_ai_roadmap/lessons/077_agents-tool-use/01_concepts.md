# 01 — Concepts: Agents & Tool Use

## The core limitation this solves

An LLM is fundamentally a text predictor (Lesson 063) — it cannot
literally look up today's weather, execute code, or query a database. It
*can*, however, be trained/prompted to recognize when a task needs an
external capability, and to **generate a structured request** for that
capability — with the actual execution happening outside the model, in
ordinary code, and the result fed back in as additional context.

## Function/tool calling: the basic mechanism

```python
tools = [{
    "name": "get_weather",
    "description": "Get the current weather for a city",
    "parameters": {"city": "string"},
}]

response = llm.generate(prompt, tools=tools)
# model outputs structured JSON instead of (or alongside) free text:
# {"tool_call": {"name": "get_weather", "arguments": {"city": "Paris"}}}

if response.tool_call:
    result = execute_tool(response.tool_call)          # YOUR code actually runs this
    followup_prompt = prompt + f"\nTool result: {result}"
    final_response = llm.generate(followup_prompt)       # model incorporates the real result
```

The model never "executes" anything itself — it predicts (Lesson 063,
unchanged) a structured request; your surrounding code is responsible for
actually calling the real function/API and feeding the result back as
more context for a follow-up generation. This is exactly the same
"generate → external system responds → generate again with new context"
loop as RAG (Lesson 076), generalized from "retrieve documents" to
"call arbitrary tools."

## The ReAct pattern: interleaving reasoning and action

**ReAct** (Reasoning + Acting) prompts the model to explicitly alternate
between reasoning steps and tool calls, in a visible trace:

```
Thought: I need to know the population of France to answer this.
Action: search("population of France")
Observation: France's population is approximately 68 million (as of 2023).
Thought: Now I can answer the original question.
Answer: France has approximately 68 million people.
```

Making the reasoning **explicit** (as generated text, not hidden) tends to
improve task performance — the model effectively "thinks out loud" through
each step, and each subsequent generation is conditioned on that visible
reasoning trace (this connects directly to Lesson 063's "self-supervised,
predict-what-comes-next" framing — a visible reasoning trace becomes part
of the context future tokens attend to, via Lesson 058's attention,
directly shaping what gets generated next).

## Agent loops: repeating until the task is done

```python
def run_agent(task, tools, max_steps=10):
    context = f"Task: {task}\n"
    for step in range(max_steps):
        response = llm.generate(context)
        if response.is_final_answer:
            return response.answer
        result = execute_tool(response.tool_call)
        context += f"\nAction: {response.tool_call}\nObservation: {result}\n"
    return "Max steps reached without a final answer."
```

This is Lesson 066's generation loop, extended: instead of generating
tokens until an end-of-sequence marker, the agent loop generates until
either a final answer or a step budget is exhausted, with each iteration
potentially expanding the context via a real tool call's result.

## Multi-agent systems (a brief mention)

Some applications coordinate **multiple** LLM instances with different
roles/prompts (e.g. a "planner" agent, a "coder" agent, a "reviewer"
agent) passing outputs between each other — the same underlying mechanism
(structured generation, tool calls, context accumulation) as a single
agent, just orchestrated across several model calls with different system
prompts/roles rather than one continuous loop.

## What can go wrong: real, documented failure modes

- **Tool call hallucination**: the model requests a tool that doesn't
  exist, or with malformed/nonsensical arguments — real systems need
  validation before actually executing anything the model requests.
- **Infinite or unproductive loops**: an agent can get stuck
  retrying a failing action repeatedly — step budgets (`max_steps` above)
  and loop-detection are standard, necessary safeguards.
- **Cascading errors**: a wrong tool result early in a multi-step task can
  propagate and compound through subsequent reasoning steps, since each
  step's context includes everything before it.
- **Security**: any tool that executes code, makes purchases, sends
  messages, or modifies data on the model's initiative needs careful
  scoping/sandboxing/human-approval steps — treating raw LLM tool-call
  output as safe-to-execute-unchecked is a real, serious risk in
  production agent systems (directly relevant to this repo's own
  human-in-the-loop confirmation norms for risky actions).

## Where this leaves the curriculum

Lesson 076 (RAG) and this lesson (agents/tools) are the two dominant
patterns for making an LLM (yours, from Module 11, or a commercial one)
genuinely useful beyond raw text completion — both are "generate, consult
something external, generate again" loops, differing in what the external
thing is (a document store vs. an arbitrary callable tool). Project 014
builds a complete RAG system directly; understanding this lesson's tool-
calling pattern equips you to extend it (or any LLM application) with
additional live capabilities beyond retrieval alone.
