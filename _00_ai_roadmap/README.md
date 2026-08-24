# AI/ML Learning Repo

Goal: go from AI/ML basics to being able to build my own LLM from scratch —
and use it to switch careers from full-stack web development (Java/Node +
React/Angular) into an AI/ML developer role. See [career-transition.md](career-transition.md)
for that side of the plan (portfolio milestones, interview prep, reference links).

## Repo Structure

```text
roadmap.md          Master lesson/project/assignment index (source of truth for sequencing)
PROGRESS.md          My personal completion checklist
GLOSSARY.md          Running glossary of terms, added to as lessons introduce them
requirements.txt     Shared Python environment for all lessons/projects

lessons/NNN_slug/           One focused concept each
  README.md                 Objective, prerequisites, estimated time
  01_concepts.md             Theory + math
  02_practicals.md           Hands-on exercises (no answers)
  03_solutions.md             Worked solutions to the practicals

projects/NNN_slug/           Real-world-flavored builds, referencing prerequisite lessons
  README.md
  01_requirement.md           Problem statement / spec
  02_solutions/                Runnable code + data/ subfolder

assignments/NNN_slug/        Checkpoint quizzes/exercises after a block of lessons
  README.md
  01_questions.md
  02_solutions/                Runnable code + explanations
```

Numbering for `lessons/`, `projects/`, and `assignments/` is independent per folder —
a project's README always states which lesson numbers it assumes.

Start here: [roadmap.md](roadmap.md).

## Road Map Reference

1. [Scaler](https://www.scaler.com/blog/machine-learning-roadmap/)
2. [geeksforgeeks](https://www.geeksforgeeks.org/blogs/machine-learning-roadmap/) ***
3. [roadmap.sh](https://roadmap.sh/ai-engineer)
4. [AI ML Road Map](https://www.youtube.com/watch?v=0v0GZjLBUYk&list=PLlpUUtQ9RrF76jvALwrTp0oOGfk0EGC3s&index=33)
5. [Youtube & Github](https://github.com/krishnaik06/Perfect-Roadmap-To-Learn-Data-Science-In-2025) ***

## Path

### 1. Mathematical and Theoretical

1. Probability
2. Statistics
3. Linear Algebra
4. Calculus
5. Gradient Descent
6. Bias-Variance Trade-Off
7. Evaluation
8. ML Algorithms
9. Curse of Dimensionality
10. Neural Networks

---

### 2. Programming and Tools

1. Python
2. Data Structures
3. NumPy
4. Pandas
5. Plotting Libs ( Matplotlib )

## References

### Math

1. [Math For AI/ML](./roadmap.md)
2. [AI-ML-Roadmap-from-scratch - Github](https://github.com/aadi1011/AI-ML-Roadmap-from-scratch?tab=readme-ov-file#module-0---before-you-start)
3. [Introduction to Machine Learning](https://www.youtube.com/playlist?list=PLD80i8An1OEHSai9cf-Ip-QReOVW76PlB)
4. [Building a neural network FROM SCRATCH](https://www.youtube.com/watch?v=w8yWXqWQYmU)
5. [The Elegant Math Behind Machine Learning](https://www.youtube.com/watch?v=URtF_UHYBSo)
6. [Harvard CS50’s Artificial Intelligence with Python – Full University Course](https://www.youtube.com/watch?v=5NgNicANyqM)
7. [Machine Learning with Python and Scikit-Learn – Full Course](https://www.youtube.com/watch?v=hDKCxebp88A)

#### Probability Reference

- [Probability - JensenMath](https://www.youtube.com/watch?v=LgLgexX7iTs)

#### Statistics

- [Statistics - Dr. Abhinanda Sarkar](https://www.youtube.com/watch?v=Vfo5le26IhY)
- [Statistics - A Full Lecture to learn Data Science (2025 Version)](https://www.youtube.com/watch?v=K9teElePNkk)

#### Andrej Karpathy — Build Your Own LLM

The primary inspiration for [roadmap.md](roadmap.md)'s Module 11 (Building
Your Own LLM): everything built from scratch in plain PyTorch, starting
character-level, before touching any pretrained model or high-level library.

- [Neural Networks: Zero to Hero (playlist)](https://karpathy.ai/zero-to-hero.html) — micrograd (autograd/backprop from scratch) → makemore (bigram → MLP → WaveNet-style char-level language models)
- [micrograd (repo)](https://github.com/karpathy/micrograd) — tiny autograd engine, the conceptual root of lesson 038 (NN from scratch)
- [makemore (repo)](https://github.com/karpathy/makemore) — character-level language modeling, warm-up for lesson 063a
- [Let's build GPT: from scratch, in code, spelled out (video)](https://www.youtube.com/watch?v=kCc8FmEb1nY) — lessons 064–066 follow this directly
- [nanoGPT (repo)](https://github.com/karpathy/nanoGPT) — the reference implementation Project 013 is modeled on
- [minbpe (repo)](https://github.com/karpathy/minbpe) — BPE tokenizer from scratch, lesson 068a
- [Let's reproduce GPT-2 (video)](https://www.youtube.com/watch?v=l8pRSuU81PU) — scaling up, lessons 067–068

#### Overall Maths

- [JensenMath](https://www.youtube.com/@MrJensenMath10)
- [My Lesson](https://www.youtube.com/watch?v=0z6AhrOSrRs)
- [3Blue1Brown](https://www.youtube.com/@3blue1brown)
- [Data Dissection](https://www.youtube.com/playlist?list=PLlpUUtQ9RrF5yZ0gaUiTZ6kogQnb74Lt3)
- [Mathematics for Machine Learning Specialization](https://www.coursera.org/specializations/mathematics-machine-learning)
  - [Mathematics for Machine Learning and Data Science Specialization](https://www.coursera.org/specializations/mathematics-for-machine-learning-and-data-science)
  - [Maths for Machine Learning](https://www.youtube.com/playlist?list=PLD80i8An1OEGZ2tYimemzwC3xqkU0jKUg)

## My Approach

- [Probability - JensenMath](https://www.youtube.com/watch?v=LgLgexX7iTs)
- [Statistics](https://www.youtube.com/watch?v=Vfo5le26IhY) - whatched only statistic part of the video
- [Entire Maths](https://www.youtube.com/playlist?list=PLlpUUtQ9RrF5yZ0gaUiTZ6kogQnb74Lt3)

## See This

- [Check](https://www.youtube.com/@NeuralNine/playlists)
  - [ML](https://www.youtube.com/watch?v=jg5paDArl3E&list=PL7yh-TELLS1EZGz1-VDltwdwZvPV-jliQ) ***

- [Check](https://www.youtube.com/@krishnaik06)                                       *** some are paid
  - [ML](https://www.youtube.com/watch?v=bPrmA1SEN2k&list=PLZoTAELRMXVPBTrWtJkn3wWQxZkmTXGwe)
  - [ML](https://www.youtube.com/watch?v=JxgmHe2NyeY)  ***

- [Check-ML](https://www.youtube.com/watch?v=rLOyrWV8gmA)

- [Jovian](https://www.youtube.com/@jovianhq/playlists)                               *** some are paid
  - [Full Video](https://www.youtube.com/watch?v=hDKCxebp88A&t=91s)

## Next To Watch

- [ML](https://www.youtube.com/watch?v=trsyTEA22Gw&list=PLlpUUtQ9RrF5yZ0gaUiTZ6kogQnb74Lt3&index=6)
- [ML](https://www.youtube.com/watch?v=i_LwzRVP7bg)
- [ML](https://www.youtube.com/watch?v=JxgmHe2NyeY) ***
- [9th Grade Student - ML](https://www.youtube.com/watch?v=NWONeJKn6kc)
      [Check Full](https://www.youtube.com/@Neweraa/playlists)

##

- [Kylie Ying](https://www.youtube.com/@KylieYYing)
  - [ML - Maths](https://www.youtube.com/watch?v=bk12t0Xz5FM&list=PLkWv3oO4kHnu095L52vLCVK8YN33yRrd_)
  - [ML](https://www.youtube.com/watch?v=i_LwzRVP7bg)

## Do This

- watch this to revision - 27 Min
   [Essential Machine Learning and AI Concepts Animated](https://www.youtube.com/watch?v=PcbuKRNtCUc)

- Algos - 34 Min
   [ML Algos](https://www.youtube.com/watch?v=BUTjcAjfMgY)

- AI Model revision
   [All Machine Learning Models Clearly Explained!](https://www.youtube.com/watch?v=0YdpwSYMY6I)

- Check this for individual concepts
  [AI For Beginners](https://www.youtube.com/@EasyAIForAll/videos)

- [Freecode Camp - ML](https://www.youtube.com/watch?v=i_LwzRVP7bg&list=PLWKjhJtqVAblStefaz_YOVpDWqcRScc2s)

## Full Course

- [Siddhardhan](https://www.youtube.com/watch?v=LcWFedjaR4Q&list=PLfFghEzKVmjvII5ZcBnFWQOUjtUVdDnmo)
- [AI - Harvard CS50’s](https://www.youtube.com/watch?v=5NgNicANyqM&t=91s)
- [codebasics](https://www.youtube.com/watch?v=gmvvaobm7eQ&list=PLeo1K3hjS3uvCeTYTeyfe0-rN5r8zn9rw) ****
