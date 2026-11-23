# Aid Impact Simulator: Reinforcement Learning for NGO Aid Allocation

## Mission

Many NGOs give help that fills today's gap, while the same problem comes back tomorrow.
This project trains reinforcement learning agents to decide **what support to give, to whom
and when**, so that households move toward lasting stability instead of depending on handouts.

## The environment

The agent is an NGO program officer running a two year program (104 weeks) for 12 vulnerable
households with a limited budget. Each week one household is reviewed and the agent picks one action.

| # | Action | Cost | Effect |
|---|---|---|---|
| 0 | Monitor only | 0 | Nothing happens, records get older |
| 1 | Cash transfer | 2 | Food security up now, aid dependency up |
| 2 | Give productive asset | 8 | Income grows, but only after 8 weeks |
| 3 | Skills training | 4 | Asset becomes more productive |
| 4 | Health support | 3 | Health up, illness does less damage |
| 5 | Join savings group | 1 | Savings grow and soften shocks |
| 6 | Assessment visit | 0.5 | Refreshes the household record |
| 7 | Graduate household | 0 | Household leaves the program |

Random droughts and illness hit households during the program. A household graduated too
early can fall back into crisis, which gives a penalty. A graduation that holds for 12 weeks
gives a bonus.

## How to run

This project uses [uv](https://docs.astral.sh/uv/) for everything.

```bash
uv sync
uv run main.py
```

`main.py` starts a local server and opens the 3D village (three.js) in your browser at
http://127.0.0.1:8000. The terminal prints a verbose log of every week. An internet
connection is needed the first time so the browser can load three.js from the CDN.

Other commands:

```bash
uv run main.py --text                          # terminal only, no 3D view
uv run play.py                                 # same as main.py
uv run python -m training.evaluate_baselines   # compare hand-made policies
uv run pytest                                  # run the tests
```

## Project structure

```
environment/   custom environment and visualization
training/      training scripts for DQN and the policy gradient methods
models/        saved models
logs/          training logs and results
assets/        figures and images for the report
tests/         environment tests
```
