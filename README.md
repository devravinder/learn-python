# Python

## Pyhthon Interpreter

1. to enter Pyhthon Interpreter just type `python3` in cmd

---

## ✅ Python Learning Roadmap — Full List

1. Hello World → printing, comments
2. Variables & Data Types → int, float, string, bool, list, tuple, dict, set
3. Operators → arithmetic, comparison, logical, assignment, identity, membership
4. Type Casting & Input → `int()`, `str()`, `input()`
5. Strings → slicing, formatting, methods
6. Conditional Statements → if, if-else, elif, nested if
7. Loops → for, while, break, continue, range
8. Functions → parameters, default params, return
9. Modules & Packages → import, create your own module
10. Exception Handling → try, except, finally, raise
11. List & Tuple → CRUD operations, comprehension
12. Dictionary & Set → CRUD, methods
13. File Handling → read, write, append
14. Classes & Objects → constructor, methods
15. OOP Principles → inheritance, encapsulation, polymorphism
16. Magic / Dunder Methods → `__str__`, `__len__`, etc
17. Decorators → function & class decorators
18. Generators → `yield`
19. Iterators → `__iter__`, `__next__`
20. Lambda & Map / Filter / Reduce → functional programming
21. Virtual Environments → venv, pip
22. Testing → unittest / pytest
23. Async Programming → async / await
24. Web Frameworks → Flask, FastAPI

---

## Naming Convention

---

## Editor Setup

### VSC

1. install extensions

   * python
   * autopep8 (for formatting)

2. shortcuts

   * ctrl + k, ctrl + f = formatting (selection)

3. add run short key `Ctrl + R, Ctrl + P`

   * go to keyboard shortcuts (Ctrl + K, Ctrl + S)
   * or Ctrl + Shift + P → keyboard shortcuts
   * search for "run python file"
   * add key binding `Ctrl + R, Ctrl + P`

   Note:

   * we can add key binding for any programming language
   * example for node.js:

     1. goto keyboard shortcuts (Ctrl + K, Ctrl + S)
     2. click on json symbol (keybindings.json)
     3. add the below code:

     ```json
     {
       "key": "ctrl+r ctrl+n",
       "command": "workbench.action.terminal.sendSequence",
       "args": { "text": "node ${file}\u000D" }
     }
     ```

     `\u000D` means Enter key automatically.

---

## Required Tools & Packages

1. Python3

2. VSC

3. pip

4. pipx

   ```bash
   sudo apt install pipx
   pipx ensurepath
   ```

5. python3.12-venv (for virtual env)

6. poetry

   ```bash
   pipx install poetry
   ```

7. uv

   ```bash
    pipx install uv
   ```
