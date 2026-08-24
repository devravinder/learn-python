# 03 — Solutions: Agents & Tool Use

*(This code was actually run to produce the numbers below.)*

## 1–3. The agent loop, working end to end

```python
import re

def calculator(expression):
    if not re.fullmatch(r"[\d\s\+\-\*/\.]+", expression):
        raise ValueError("unsafe expression")
    return eval(expression, {"__builtins__": {}}, {})   # restricted eval - digits/operators only

def fake_llm(context):
    match = re.search(r"(\d+\s*[\+\-\*/]\s*\d+(?:\s*[\+\-\*/]\s*\d+)*)", context)
    if match and "Observation:" not in context:
        return {"tool_call": {"name": "calculator", "arguments": {"expression": match.group(1)}}}
    obs_match = re.findall(r"Observation:\s*([\-\d\.]+)", context)
    if obs_match:
        return {"final_answer": f"The answer is {obs_match[-1]}."}
    return {"final_answer": "I could not determine an answer."}

TOOLS = {"calculator": calculator}

def run_agent(task, tools, max_steps=10):
    context = f"Task: {task}\n"
    for step in range(max_steps):
        response = fake_llm(context)
        if "final_answer" in response:
            return response["final_answer"], step
        call = response["tool_call"]
        name = call["name"]
        if name not in tools:
            context += f"\nObservation: ERROR - tool '{name}' does not exist\n"
            continue
        result = tools[name](**call["arguments"])
        context += f"\nAction: {name}({call['arguments']})\nObservation: {result}\n"
    return "Max steps reached without a final answer.", max_steps

answer, steps = run_agent("What is 47 * 12 + 8?", TOOLS)
print(answer, steps)
```

**Actual output: `The answer is 572.`, `steps=1`** — matching the correct
arithmetic (`47*12+8 = 572`) exactly, via the full loop: the fake model
requested the calculator tool with the extracted expression, the tool
actually executed it, the result was appended to context as an
`Observation:`, and the next model call read that observation to produce
the final answer — precisely `01_concepts.md`'s described mechanism,
verified end to end.

## 4. `max_steps` safeguard against infinite loops

```python
def buggy_fake_llm(context):
    match = re.search(r"(\d+\s*[\+\-\*/]\s*\d+(?:\s*[\+\-\*/]\s*\d+)*)", context)
    return {"tool_call": {"name": "calculator", "arguments": {"expression": match.group(1)}}}
    # BUG: never checks for an existing Observation, always requests the tool again

def run_agent_buggy(task, tools, llm_fn, max_steps=5):
    context = f"Task: {task}\n"
    for step in range(max_steps):
        response = llm_fn(context)
        if "final_answer" in response:
            return response["final_answer"], step
        call = response["tool_call"]
        result = tools[call["name"]](**call["arguments"])
        context += f"\nObservation: {result}\n"
    return "Max steps reached without a final answer.", max_steps

print(run_agent_buggy("What is 5 + 5?", TOOLS, buggy_fake_llm, max_steps=5))
```

**Actual output: `('Max steps reached without a final answer.', 5)`.** The
buggy "model" would loop forever (always re-requesting the tool, never
noticing the observation), but `max_steps` correctly cuts it off at
exactly 5 iterations instead of hanging — confirming the safeguard from
`01_concepts.md` does its job.

## 5. Handling a hallucinated tool call gracefully

```python
def hallucinating_llm(context):
    if "Observation:" not in context:
        return {"tool_call": {"name": "web_search", "arguments": {"query": "foo"}}}   # not a real tool!
    return {"final_answer": "done"}

def run_agent_with_validation(task, tools, llm_fn, max_steps=10):
    context = f"Task: {task}\n"
    for step in range(max_steps):
        response = llm_fn(context)
        if "final_answer" in response:
            return response["final_answer"], step
        call = response["tool_call"]
        name = call["name"]
        if name not in tools:
            context += f"\nObservation: ERROR - tool '{name}' does not exist\n"
            continue
        result = tools[name](**call["arguments"])
        context += f"\nObservation: {result}\n"
    return "Max steps reached without a final answer.", max_steps

print(run_agent_with_validation("task", TOOLS, hallucinating_llm, max_steps=3))
```

**Actual output: `('done', 1)`.** Requesting the nonexistent `"web_search"`
tool didn't crash the loop with a raw `KeyError` — the validation check
caught it, recorded an error observation, and the loop continued
gracefully (in this toy example, the fake model then produced a final
answer on the next step). Real agent systems apply exactly this kind of
validation before ever executing a model-requested tool call.

## 6. Where a human-approval step would go

A human-approval checkpoint would insert **right after a tool call is
requested and validated, but before it's actually executed** — precisely
the point in `run_agent`'s loop between `call = response["tool_call"]` and
`result = tools[name](**call["arguments"])`. For a consequential action
(sending an email, making a purchase, deleting a file), the loop would
pause there, present the human with: the original task/context, the exact
tool and arguments about to be executed, and (if available) a plain-
language description of the expected effect — then wait for an explicit
approve/reject response before either proceeding with execution or
substituting a rejection message as the "observation" and letting the
agent adapt its plan accordingly. This is exactly the same "transparently
communicate the risky action and get confirmation before proceeding"
principle this repo's own working norms already follow for actions like
`git push` or destructive file operations — the same idea, generalized to
any AI agent taking real-world actions, not just this specific
development context.
