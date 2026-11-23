# Aid Impact Simulator: Reinforcement Learning for NGO Aid Allocation

## Mission

Many NGOs give help that fills today's gap, while the same problem comes back tomorrow.
This project trains reinforcement learning agents to decide **what support to give, to whom
and when**, so that households move toward lasting stability instead of depending on handouts.

## The environment

The agent is an NGO program officer running a two year program (104 weeks) for 12 vulnerable
households with a budget of 120 units. Each week one household is reviewed and the agent picks one action.

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

**Observation:** 16 numbers between 0 and 1. Six come from the officer's notebook record of the
household (food, income, savings, health, school, aid dependency), which can be weeks old. The rest are
program flags (asset, training, savings group, health cover) and program state (record age, asset age,
budget left, weeks left, drought, share graduated).

**Reward:** +10 x the change in the average "lasting score" of all households, -0.3 per household in
crisis each week, +5 when a graduation holds for 12 weeks, -2 when a household is not ready at its check,
-5 when a graduated household falls back into crisis, and +3 per household on track at the end.

**End of episode:** when every household has graduated and passed its check, or at week 104.
Random droughts and illness hit households during the program.

## How to run

This project uses [uv](https://docs.astral.sh/uv/) for everything.

```bash
uv sync
uv run main.py
```

`main.py` starts a local server and opens the 3D village (three.js) in your browser at
http://127.0.0.1:8000. The terminal prints a verbose log of every week. The policy menu lists the
best trained agent of each algorithm (DQN, REINFORCE, PPO, A2C) followed by the hand-made baselines.
three.js is included in the project (`environment/web/vendor/`, MIT license), so the 3D view works offline.

Other commands:

```bash
uv run main.py --text                          # terminal only, no 3D view
uv run main.py --policy ppo_agent              # start with another agent selected
uv run play.py                                 # same as main.py
uv run python -m training.evaluate_baselines   # compare hand-made policies
uv run pytest                                  # run the tests
```

The server also has an API: `GET /api/info` and `POST /api/recommend`, which takes 16 observation
numbers and a policy name and returns the recommended action.

## Training

Every algorithm is trained with TorchRL for 150,000 environment steps, then tested on the same
20 programs (seeds 5000 to 5019).

```bash
uv run python -m training.dqn_training --list                    # show the DQN experiments
uv run python -m training.dqn_training --run dqn_01_baseline     # train one DQN run
uv run python -m training.pg_training --algo ppo --run ppo_01_baseline
uv run python -m training.run_sweep --workers 4                  # run every experiment
uv run python -m training.analysis                               # pick best models, make figures
```

## Results

53 hyperparameter runs in total: DQN 17, REINFORCE 12, PPO 13, A2C 11.

| Agent | Best run | Mean reward | Graduations that held | Households in crisis at end |
|---|---|---|---|---|
| Rule-based officer (baseline) | | 8.5 | 0.5 | 0.0 |
| REINFORCE | reinforce_09_high_entropy | -6.0 | 0.0 | 0.0 |
| PPO | ppo_10_lambda_100 | -8.7 | 0.0 | 0.0 |
| DQN | dqn_17_reward_scale | -21.4 | 5.8 | 0.8 |
| A2C | a2c_01_baseline | -33.1 | 0.0 | 2.2 |
| Cash-only officer (baseline) | | -146.9 | 0.0 | 11.9 |
| Random officer (baseline) | | -202.7 | | |

What the agents learned:

- **PPO and REINFORCE** give each household a productive asset, then only monitor. Nobody stays
  in crisis, but nobody graduates.
- **DQN** builds income and savings, then graduates households. It is the only agent whose
  graduations last, but it is less stable and suffers more when droughts are frequent.
- **A2C** learned to give cash transfers: food looks fine today, but no lasting income is built.
- Runs with a short planning horizon (gamma 0.90) fell back to cash every time.

Figures are in `assets/`. Their numbers match the figures in the report:

| File | What it shows |
|---|---|
| `fig1_environment.png` | the 3D village during a program |
| `fig2_rewards.png` | reward during training, best run of each algorithm |
| `fig3_behaviour.png` | which actions each agent chooses |
| `fig4_training_stability.png` | DQN loss and policy entropy during training |
| `fig5_generalization.png` | best agents on unseen and harsher programs |
| `extra_sweeps.png` | every hyperparameter run (extra, not in the report) |

Full tables are in `logs/tables/`.

## Project structure

```
environment/   custom environment and 3D visualization (three.js)
training/      training scripts for DQN, REINFORCE, PPO and A2C, analysis
models/        saved models (models/dqn, models/pg); best_*.pt is the best run per algorithm
logs/          training logs, results of every run and summary tables
assets/        figures for the report
tests/         environment tests
main.py        entry point (play.py does the same)
```
