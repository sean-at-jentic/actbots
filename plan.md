### Project Goal

Build a **clean, reusable Python library** that lets any developer spin up AI agents whose reasoning loops automatically *search → load → execute* Jentic workflows and API operations.
The library must:

1. Encapsulate the evolving “state-of-the-art” reasoning loop in a **StandardReasoner** that can be swapped out later without breaking public interfaces.
2. Provide a **BaseAgent** that composes this reasoner with memory, an inbox (where goals arrive), and a thin Jentic client.
3. Keep all dependencies isolated inside a project-local virtual environment so nothing leaks into the host Python install.

### How the design achieves the goal

| Layer                          | What it does                                                                                                                                                                                                                     | Why it matters                                                                                        |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **reasoners/**                 | `BaseReasoner` (abstract) defines *plan → select\_tool → act → observe → evaluate (+reflect)* hooks. `StandardReasoner` implements those hooks using ReAct + self-reflection **and calls Jentic SDK** to discover and run tools. | Lets you upgrade the loop (e.g., switch to Reflexion) by adding a new subclass, not rewriting agents. |
| **platform/jentic\_client.py** | Thin adapter over `jentic-sdk` that exposes `search`, `load`, `execute`.                                                                                                                                                         | Centralises auth, retries, logging, and keeps the rest of the codebase SDK-agnostic.                  |
| **agents/**                    | `BaseAgent` wires together reasoner, memory, inbox and Jentic client; concrete agents (e.g., `InteractiveCLIAgent`) override I/O only.                                                                                           | Down-stream devs can create new front-ends (Slack, REST) without touching reasoning logic.            |
| **memory/**                    | Plug-and-play memory back-ends (start with simple scratch-pad).                                                                                                                                                                  | Supports future upgrades to vector DB or long-term skill libraries.                                   |
| **inbox/**                     | Streams goals into the agent (CLI, message queue, cron job…).                                                                                                                                                                    | Decouples goal-source from agent internals; easy to add new integrations.                             |
| **tests/**                     | Unit-test every contract and mock Jentic SDK; optional integration test with a real key.                                                                                                                                         | Ensures that refactors or new reasoning patterns do not break core behaviour.                         |

### Key implementation decisions the agent should respect

* **Strict interfaces first** – abstract base classes (ABCs) with type hints give compile-time guarantees and guide future contributors.
* **Dependency isolation** – always create and use `.venv`; install from `requirements.txt`; no global installs.
* **Single source of truth for Jentic calls** – only `JenticClient` contacts the SDK; everyone else depends on that wrapper.
* **Stateless reasoning contract** – `BaseReasoner.run()` returns a packaged result object; side-effects (memory writes, inbox acks) are handled by the agent layer, not the reasoner.
* **Fail fast, log clear** – raise explicit exceptions on tool failure or budget exhaustion; include enough context for debugging.
* **Testability** – design every external call (LLM, Jentic) behind an injectable callable so unit tests can mock them.
* **Extensibility over cleverness** – prefer simple, readable code with docstrings and mypy-friendly types to fancy metaprogramming.

With this context the implementation agent should be able to make sound choices—e.g., where to locate new files, how to name classes, how to wire mocks—while staying perfectly aligned with the project’s purpose: **a modular, future-proof foundation for Jentic-powered autonomous agents.**

Below is a **turn-key execution plan** that you can hand to an autonomous coding agent (or follow yourself) to build the project from scratch while keeping the host machine pristine.

---

## 1 Environment & dependency bootstrap

| Step    | Command                               | Notes                                            |
| ------- | ------------------------------------- | ------------------------------------------------ |
| **1.1** | `python3 -m venv .venv`               | project-local virtual environment                |
| **1.2** | `source .venv/bin/activate`           | activate it (use `Scripts\\activate` on Windows) |
| **1.3** | `python -m pip install --upgrade pip` | always start clean                               |
| **1.4** | create `requirements.txt` (see below) | freeze deps for reproducibility                  |
| **1.5** | `pip install -r requirements.txt`     | pulls the Jentic SDK + dev tools                 |

`requirements.txt`

```
# core
jentic-sdk>=0.1.0
openai>=1.0
pydantic>=2.0

# dev / quality
pytest>=8.0
pytest-mock
ruff            # fast linter/formatter
mypy
```

*(Optional)* add a `Makefile` helper:

```make
install:          ## create venv + install deps
	python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
test:             ## run unit tests
	. .venv/bin/activate && pytest -q
lint:             ## static checks
	. .venv/bin/activate && ruff check . && mypy .
```

---

## 2 Project skeleton

```
jentic_agents/
│
├─ reasoners/
│   ├─ base_reasoner.py          # abstract contract
│   └─ standard_reasoner.py      # ReAct + Jentic SDK (already drafted)
│
├─ agents/
│   ├─ base_agent.py
│   └─ interactive_cli_agent.py
│
├─ memory/
│   ├─ base_memory.py
│   ├─ scratch_pad.py
│   └─ vector_memory.py          # (optional, future)
│
├─ inbox/
│   ├─ base_inbox.py
│   └─ cli_inbox.py              # pulls goals from stdin
│
├─ platform/                     # thin wrapper for Jentic SDK
│   └─ jentic_client.py
│
├─ tests/
│   ├─ test_reasoner.py
│   ├─ test_agent_cli.py
│   └─ fixtures.py
│
├─ requirements.txt
└─ README.md
```

---

## 3 Object-model recap

| Layer         | Class                  | Responsibility                                                          |
| ------------- | ---------------------- | ----------------------------------------------------------------------- |
| **Reasoning** | `BaseReasoner` *(ABC)* | Defines `plan`, `select_tool`, `act`, `observe`, `evaluate`, `reflect`. |
|               | `StandardReasoner`     | Concrete ReAct loop wired to **search → load → run** via Jentic SDK.    |
| **Agent**     | `BaseAgent` *(ABC)*    | Composes reasoner, memory, inbox, Jentic client; provides `spin()`.     |
|               | `InteractiveCLIAgent`  | Reads goals from CLI inbox, streams answers to stdout.                  |
| **Memory**    | `BaseMemory` *(ABC)*   | Simple `store` / `retrieve`. Subclass: `ScratchPadMemory`.              |
| **Inbox**     | `BaseInbox` *(ABC)*    | Yields goals; subclass `CLIInbox`.                                      |
| **Platform**  | `JenticClient`         | Thin adapter around `jentic_sdk` (auth, retries, logging).              |

---

## 4 Step-by-step implementation roadmap

| Phase                      | What to do                                                                                                                                        | Acceptance test(s)                                                                                                                                                                 |                                                                                      |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| **P0 repo setup**          | - Initialize Git repo, add skeleton tree above.<br>- Commit `requirements.txt`, `README.md`.                                                      | `pytest` runs with zero tests, exits 0.                                                                                                                                            |                                                                                      |
| **P1 platform wrapper**    | Implement `platform/jentic_client.py`:<br>\`\`\`python\nclass JenticClient:\n    def **init**(self, api\_key: str                                 | None = None): ...\n    def search(self, query: str, top\_k: int = 5): ...\n    def load(self, tool\_id: str): ...\n    def execute(self, tool\_id: str, params: dict): ...\n\`\`\` | Unit test uses `pytest-mock` to patch SDK calls; all three methods return stub data. |
| **P2 core abstractions**   | Write `BaseReasoner`, `BaseAgent`, `BaseMemory`, `BaseInbox` exactly as designed earlier (ABC + docstrings).                                      | `mypy` passes; classes cannot be instantiated directly.                                                                                                                            |                                                                                      |
| **P3 StandardReasoner**    | Paste the fully drafted file from the canvas; adjust import paths.                                                                                | `tests/test_reasoner.py` mocks `JenticClient` and OpenAI, asserts that `StandardReasoner.run()` returns a `final_answer` for a toy goal in ≤ N iterations.                         |                                                                                      |
| **P4 supporting pieces**   | Implement `ScratchPadMemory` (dict-based) and `CLIInbox`.                                                                                         | Tests verify that goals typed in stdin reach the reasoner.                                                                                                                         |                                                                                      |
| **P5 InteractiveCLIAgent** | Combine all parts; graceful exit on “quit”.                                                                                                       | Manual smoke test: run `python -m agents.interactive_cli_agent`, ask “What’s 2+2?”.                                                                                                |                                                                                      |
| **P6 lint & type gate**    | Configure **ruff** and **mypy.ini**; update `Makefile`.                                                                                           | `make lint` exits 0.                                                                                                                                                               |                                                                                      |
| **P7 integration test**    | If you have sandbox creds, allow `test_agent_cli.py::test_end_to_end` to hit the real Jentic search/echo tool. Otherwise skip with `pytest.skip`. | Test prints real workflow output.                                                                                                                                                  |                                                                                      |
| **P8 Docs**                | Flesh out `README.md`: setup, quickstart, architecture diagram (mermaid), contribution guide.                                                     | Markdown renders cleanly on GitHub.                                                                                                                                                |                                                                                      |
| **P9 CI (optional)**       | Add GitHub Actions: `setup-python`, `make install`, `make lint`, `pytest`.                                                                        | PR badge shows ✅.                                                                                                                                                                  |                                                                                      |

---

## 5 Testing criteria (definition of done)

1. **Unit tests**

   * > 90 % coverage on `reasoners/` and `agents/`.
   * Error paths (tool failure, budget exhaustion) trigger correct exceptions.

2. **Static quality**

   * `ruff` → no warnings.
   * `mypy` strict → no type errors.

3. **Integration**

   * With a valid `JENTIC_API_KEY`, `InteractiveCLIAgent` can ask Jentic for the *“ping”* workflow, execute it, and echo the reply.

4. **Developer ergonomics**

   * One-command bootstrap: `make install`.
   * One-command lint+test: `make lint && make test`.

5. **Isolation**

   * No packages installed outside `.venv`.
   * Deleting `.venv` leaves host environment unchanged.

