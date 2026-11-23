"""
Visualization and web server for the Aid Impact Simulator.

The 3D village is drawn in the browser with three.js (files in environment/web/).
This file runs a small FastAPI server that:

  1. serves the web page,
  2. runs the environment and sends the village state to the page after every
     step (through a WebSocket),
  3. prints a verbose log of every step in the terminal,
  4. offers a simple REST API (/api/recommend) that returns a recommended
     action for a household. This shows how an NGO dashboard could use the agent.
"""

import webbrowser
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from environment.custom_env import (
    AidImpactEnv, ACTION_NAMES, ACTION_COSTS, OBSERVATION_NAMES,
    NUM_HOUSEHOLDS, PROGRAM_WEEKS, START_BUDGET, CRISIS_FOOD_LEVEL, ASSET_DELAY_WEEKS,
)

WEB_FOLDER = Path(__file__).parent / "web"


# ---------------------------------------------------------------------------
# Turning the environment into data the web page can draw
# ---------------------------------------------------------------------------
def build_snapshot(env, reward=0.0, total_reward=0.0, done=False):
    """Collect everything the 3D view needs into a plain dictionary."""
    resilience = env._resilience()
    lasting = env._lasting_score()

    households = []
    for h in range(NUM_HOUSEHOLDS):
        households.append({
            "id": h,
            # what is really happening
            "food": float(env.food[h]),
            "income": float(env.income[h]),
            "savings": float(env.savings[h]),
            "health": float(env.health[h]),
            "school": float(env.school[h]),
            "dependency": float(env.dependency[h]),
            "resilience": float(resilience[h]),
            "lasting_score": float(lasting[h]),
            # what the officer's notebook says
            "record_food": float(env.record_food[h]),
            "record_income": float(env.record_income[h]),
            "record_savings": float(env.record_savings[h]),
            "record_health": float(env.record_health[h]),
            "weeks_since_visit": int(env.weeks_since_visit[h]),
            # program register
            "has_asset": bool(env.has_asset[h]),
            "asset_ready": bool(env.has_asset[h] and env.asset_age[h] >= ASSET_DELAY_WEEKS),
            "has_training": bool(env.has_training[h]),
            "in_savings_group": bool(env.in_savings_group[h]),
            "has_health_cover": bool(env.has_health_cover[h]),
            "graduated": bool(env.graduated[h]),
            "in_crisis": bool(env.food[h] < CRISIS_FOOD_LEVEL),
        })

    return {
        "week": int(env.week),
        "program_weeks": PROGRAM_WEEKS,
        "budget": float(env.budget),
        "start_budget": START_BUDGET,
        "current_household": int(env.current),
        "last_action": None if env.last_action is None else int(env.last_action),
        "last_action_name": None if env.last_action is None else ACTION_NAMES[env.last_action],
        "reward": float(reward),
        "total_reward": float(total_reward),
        "drought": bool(env.drought_weeks_left > 0),
        "events": list(env.last_events),
        "successful_graduations": int(env.successful_graduations),
        "relapses": int(env.relapses),
        "mean_lasting_score": float(lasting.mean()),
        "households_in_crisis": int((env.food < CRISIS_FOOD_LEVEL).sum()),
        "done": bool(done),
        "households": households,
    }


def print_step(week, household, action, reward, total_reward, info):
    """Verbose terminal output for one step (needed for the video)."""
    print(f"Week {week:3d} | Household {household + 1:2d} | "
          f"{ACTION_NAMES[action]:22s} | reward {reward:7.2f} | "
          f"total {total_reward:8.2f} | budget {info['budget']:6.1f} | "
          f"crisis {info['households_in_crisis']:2d}", flush=True)
    for event in info["events"]:
        print(f"           -> {event}", flush=True)


# ---------------------------------------------------------------------------
# Web server
# ---------------------------------------------------------------------------
class HouseholdObservation(BaseModel):
    """Input for /api/recommend: the 16 observation values, all between 0 and 1."""
    observation: list[float]
    policy: str | None = None


def create_app(policies, default_policy):
    """
    policies: dictionary of name -> function(observation) -> action number
    default_policy: name of the policy selected when the page opens
    """
    app = FastAPI(title="Aid Impact Simulator")
    app.mount("/static", StaticFiles(directory=WEB_FOLDER), name="static")

    @app.get("/")
    def index():
        return FileResponse(WEB_FOLDER / "index.html")

    @app.get("/api/info")
    def info():
        return {
            "policies": list(policies.keys()),
            "default_policy": default_policy,
            "actions": [{"id": i, "name": n, "cost": c}
                        for i, (n, c) in enumerate(zip(ACTION_NAMES, ACTION_COSTS))],
            "observation_names": OBSERVATION_NAMES,
        }

    @app.post("/api/recommend")
    def recommend(body: HouseholdObservation):
        # Integration example: an NGO dashboard sends a household profile
        # and gets back the recommended intervention
        # Wrong input gets the error code 400 (bad request) and a short message
        if len(body.observation) != len(OBSERVATION_NAMES):
            raise HTTPException(400, f"observation must have {len(OBSERVATION_NAMES)} values")
        name = body.policy or default_policy
        if name not in policies:
            raise HTTPException(400, f"unknown policy '{name}', choose from: {', '.join(policies)}")
        obs = np.clip(np.array(body.observation, dtype=np.float32), 0.0, 1.0)
        action = int(policies[name](obs))
        return {"policy": name, "action": action, "action_name": ACTION_NAMES[action]}

    @app.websocket("/ws")
    async def simulation(socket: WebSocket):
        # Each browser tab gets its own environment
        await socket.accept()
        env = AidImpactEnv()
        policy_name = default_policy
        obs, step_info = env.reset(seed=0)
        total_reward = 0.0
        done = False

        try:
            while True:
                message = await socket.receive_json()

                if message["type"] == "reset":
                    policy_name = message.get("policy", default_policy)
                    seed = int(message.get("seed", 0))
                    obs, step_info = env.reset(seed=seed)
                    total_reward = 0.0
                    done = False
                    print()
                    print(f"=== New program | policy: {policy_name} | seed: {seed} ===", flush=True)
                    await socket.send_json(build_snapshot(env))

                elif message["type"] == "step" and not done:
                    household = env.current
                    action = int(policies[policy_name](obs))
                    obs, reward, terminated, truncated, step_info = env.step(action)
                    total_reward += reward
                    done = terminated or truncated
                    print_step(step_info["week"], household, action, reward, total_reward, step_info)

                    snapshot = build_snapshot(env, reward, total_reward, done)
                    snapshot["reviewed_household"] = int(household)
                    snapshot["chosen_action"] = action
                    await socket.send_json(snapshot)

                    if done:
                        print(f"=== Program finished | total reward {total_reward:.2f} | "
                              f"graduations held {step_info['successful_graduations']} | "
                              f"relapses {step_info['relapses']} | "
                              f"in crisis {step_info['households_in_crisis']} ===", flush=True)
        except WebSocketDisconnect:
            pass

    return app


def run_server(policies, default_policy, port=8000, open_browser=True):
    """Start the server and open the 3D village in the default browser."""
    app = create_app(policies, default_policy)
    url = f"http://127.0.0.1:{port}"
    print(f"Aid Impact Simulator running at {url}")
    print("Press Ctrl+C in this terminal to stop.", flush=True)
    if open_browser:
        webbrowser.open(url)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
