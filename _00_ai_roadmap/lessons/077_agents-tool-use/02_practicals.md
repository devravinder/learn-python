# 02 — Practicals: Agents & Tool Use

## The agent loop mechanics (pure Python — simulate the "model" to focus on the control flow)

Real tool-calling needs a real LLM, but the **loop structure itself** —
the part actually worth understanding deeply — can be practiced with a
simple rule-based stand-in "model," so you can test the control flow
completely deterministically before ever involving a real LLM's
unpredictability.

1. Implement a `calculator(expression)` tool: safely evaluates a simple
   arithmetic string (e.g. using `ast.literal_eval`-based parsing, or a
   restricted `eval` on digits/operators only — **never** use raw
   unrestricted `eval` on untrusted input, even in a toy exercise, as a
   matter of habit).

2. Implement a rule-based "fake LLM" `fake_llm(context)`: if the context
   contains a recognizable arithmetic expression pattern (e.g. matches a
   regex like `\d+\s*[\+\-\*/]\s*\d+`) and no `Observation:` for it yet,
   return a tool-call request for `calculator` with that expression;
   otherwise, return a final answer incorporating the most recent
   `Observation:` value from the context.

3. Implement `run_agent(task, tools, max_steps)` from `01_concepts.md`
   using your `fake_llm` and `calculator`. Run it on
   `"What is 47 * 12, plus 8?"` and confirm it correctly calls the
   calculator, incorporates the result, and produces a correct final
   answer — tracing through the full context accumulation.

4. Test the `max_steps` safeguard: modify `fake_llm` to (buggily) always
   request the calculator tool again even after receiving a valid
   observation, simulating an infinite-loop bug. Confirm `run_agent`
   correctly stops at `max_steps` instead of looping forever, and returns
   the "max steps reached" message rather than hanging.

5. Add basic tool-call validation: before executing a requested tool call,
   check that the requested tool name actually exists in your `tools`
   dict, and that required arguments are present. Simulate a "hallucinated"
   tool call (a tool name your `fake_llm` requests that isn't in your
   tools dict) and confirm your agent loop handles it gracefully (e.g.
   returns an error observation) rather than crashing with a raw
   `KeyError`.

## Reflection

6. Real production agent systems that call tools with real-world side
   effects (sending emails, making purchases, modifying files) often
   require a human-approval step before executing certain tool calls.
   Design (in words, no code needed) what such an approval step would look
   like inserted into your `run_agent` loop from Q3 — at what point in the
   loop would it need to pause, and what information would a human need to
   see to make an informed approve/reject decision?
