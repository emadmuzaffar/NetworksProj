# NetworksProj

NetworksProj is a configurable multi-agent social-relations study runner. It
assigns persona prompts to AI agents, repeatedly places every pair in a dyadic
interaction, records warmth and competence ratings, and gives each agent a
private partner-specific memory that affects later meetings.

The program is designed to run from PyCharm's **Run** window. It also has an
optional command-line entry point for automation.

## Run it from PyCharm

1. Open this repository as a PyCharm project.
2. Use Python 3.10 or newer as the project interpreter.
3. Open `main.py`.
4. Click the green triangle beside `if __name__ == "__main__"`.

No arguments or package installation are required. Progress and the output
folder appear in PyCharm's Run console.

`main.py` uses this setting near the top:

```python
CONFIG_FILE = PROJECT_ROOT / "examples" / "study.json"
```

Change that line if you make another configuration file. For a very short test,
select `examples/minimal_study.json` instead.

The included provider is deterministic and offline. It tests the entire study
loop without an API key, network connection, or model charges.

The external AI adapter may itself be a CLI program, but the study is still
started entirely through PyCharm's Run button; no commands need to be typed.

## Study loop

For each run:

1. Create the configured number of logical AI agents.
2. Assign persona prompts in a random, seeded, balanced order.
3. For every trial:
   1. Have every agent complete the personality questionnaire.
   2. Save the raw answers and persona-accuracy/consistency metrics.
   3. Run the safety checkpoint.
   4. Repeat the round robin `round_robin_repetitions` times.
   5. For every unordered pair, alternate speakers for
      `turns_per_dyad` turns.
   6. Have both agents privately rate the other agent's warmth and competence
      from 1–5.
   7. Have both agents create a private first-person memory.
   8. Add only that partner's prior memories to the agent's system prompt when
      the same dyad meets again.
4. Save differentiation metrics and directed relationship trajectories.

With `n` agents, the expected number of provider calls is:

```text
trials × agents
+ trials × (agents choose 2) × round_robin_repetitions
  × (turns_per_dyad + 2 ratings + 2 memories)
```

The default example uses 4 agents, 2 trials, 5 round robins, and 10 turns per
dyad, for 848 provider calls. Always check this number before selecting a paid
provider.

## Configuration

Edit `examples/study.json` to change:

- number of agents and trials;
- `round_robin_repetitions` (the requested `n = 5`);
- `turns_per_dyad` (the requested `k = 10`);
- interaction topic;
- questionnaire items;
- persona names, prompts, and expected questionnaire answers;
- transcript and memory limits;
- provider and runtime-safety settings.

Persona assignment is deterministic for a given `seed`, which makes a run
reproducible. When there are fewer personas than agents, the list is cycled to
keep assignments as balanced as possible, then shuffled. A persona with an
empty prompt acts as an unassigned control.

Expected questionnaire scores are optional. When present, the runner calculates
normalized persona accuracy. It also calculates:

- questionnaire consistency between agents;
- questionnaire consistency within each agent across trials;
- mean received/given warmth and competence per agent;
- between-agent standard deviation in received ratings;
- warmth and competence trajectories for every directed pair.

## Connect an AI command

The experiment is independent of a specific model vendor. To use an AI CLI or a
small adapter program, change the provider in the JSON configuration:

```json
{
  "provider": {
    "type": "command",
    "command": ["python", "my_ai_adapter.py"],
    "timeout_seconds": 120
  }
}
```

The study starts that command once for each model call and writes one JSON
object to standard input:

```json
{
  "agent_id": "agent-001",
  "system_prompt": "persona and partner-specific memories",
  "prompt": "the current task",
  "metadata": {
    "task": "interaction"
  }
}
```

The adapter must print either plain response text or one JSON envelope:

```json
{"text": "the model response"}
```

For questionnaire, rating, and memory tasks, the `text` value itself must follow
the JSON shape requested in the prompt. Each process is stateless on purpose:
the experiment runner owns persona and memory state, so different AI backends
can be compared under the same protocol.

## Runtime safety

Before the run and after each questionnaire phase, the runner checks that the
planned/actual call count is within `max_provider_calls`. It also checks before
every model call and every dyadic encounter.

To stop safely between calls, create an empty file named `STOP` inside the run's
output folder. The event log and latest state checkpoint remain readable.

For a study-specific policy check, set `safety.checker_command`. It receives a
JSON checkpoint description on standard input and must return:

```json
{"allowed": true}
```

Any provider failure, malformed structured response, call-budget violation, or
denial by the safety checker stops the run and writes the reason to disk.

## Output

Every execution creates a timestamped folder under `runs/` containing:

- `config.json` — immutable configuration snapshot;
- `events.jsonl` — append-only questionnaire, message, rating, and memory data;
- `state.json` — atomically replaced checkpoint with current agent memories;
- `summary.json` — analysis-ready differentiation and relationship metrics.

Agent IDs are pseudonymous (`agent-001`, and so on). Persona labels are stored
separately from interaction text so experimental conditions can be analyzed.

## Optional terminal use

The PyCharm Run button is the primary path. For automated tests or batch runs:

```bash
python run_study.py validate --config examples/study.json
python run_study.py run --config examples/minimal_study.json
python -m unittest discover -s tests -v
```

## Current scope

The runner creates reproducible experiment data; it does not claim that the
included questionnaire is a validated psychological instrument. Replace the
sample items and persona scoring targets with the instrument and preregistered
analysis selected for the actual study.
