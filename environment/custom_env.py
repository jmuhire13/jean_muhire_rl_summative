"""
Aid Impact Simulator

A custom Gymnasium environment based on the mission of helping NGOs
give support that creates lasting change, not only short term relief.

The agent plays the role of an NGO program officer. The program runs for
104 weeks (two years) and supports 12 vulnerable households with a limited
budget. Each week one household comes up for review and the agent picks
one action for that household.

Some actions help right away but create dependency (cash transfers).
Other actions cost more and only pay off months later (assets, training).
Droughts and illness can hit at any time. A household that is graduated
too early can fall back into crisis.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Program settings
# ---------------------------------------------------------------------------
NUM_HOUSEHOLDS = 12
PROGRAM_WEEKS = 104
START_BUDGET = 120.0

# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
MONITOR = 0
CASH_TRANSFER = 1
GIVE_ASSET = 2
SKILLS_TRAINING = 3
HEALTH_SUPPORT = 4
SAVINGS_GROUP = 5
ASSESSMENT_VISIT = 6
GRADUATE = 7

ACTION_NAMES = [
    "Monitor only",
    "Cash transfer",
    "Give productive asset",
    "Skills training",
    "Health support",
    "Join savings group",
    "Assessment visit",
    "Graduate household",
]

# Cost of each action, in budget units
ACTION_COSTS = [0.0, 2.0, 8.0, 4.0, 3.0, 1.0, 0.5, 0.0]

# ---------------------------------------------------------------------------
# Observation features (in this exact order)
# ---------------------------------------------------------------------------
OBSERVATION_NAMES = [
    "food_security",        # 0  from the household record
    "income",               # 1  from the household record
    "savings",              # 2  from the household record
    "health",               # 3  from the household record
    "school_attendance",    # 4  from the household record
    "aid_dependency",       # 5  from the household record
    "has_asset",            # 6  program register (0 or 1)
    "has_training",         # 7  program register (0 or 1)
    "in_savings_group",     # 8  program register (0 or 1)
    "has_health_cover",     # 9  program register (0 or 1)
    "weeks_since_visit",    # 10 how old the record is, scaled 0 to 1
    "asset_age",            # 11 weeks since the asset was given, scaled 0 to 1
    "budget_left",          # 12 share of the budget still available
    "weeks_left",           # 13 share of the program time still left
    "drought_active",       # 14 1 if a drought is happening now
    "share_graduated",      # 15 share of households already graduated
]

# Thresholds used by the rules of the simulation
CRISIS_FOOD_LEVEL = 0.25     # food security below this is a crisis
ASSET_DELAY_WEEKS = 8        # an asset starts paying off after this many weeks
GRADUATION_CHECK_WEEKS = 12  # graduation success is checked after this many weeks


class AidImpactEnv(gym.Env):
    """NGO aid allocation environment with delayed effects and random shocks."""

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, render_mode=None, drought_chance=0.015, illness_chance=0.015):
        super().__init__()
        self.render_mode = render_mode
        self.drought_chance = drought_chance
        self.illness_chance = illness_chance

        # 8 possible actions, one number each
        self.action_space = spaces.Discrete(len(ACTION_NAMES))

        # 16 numbers, all between 0 and 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(len(OBSERVATION_NAMES),), dtype=np.float32
        )

    # -----------------------------------------------------------------------
    # Reset: start a new two year program
    # -----------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        n = NUM_HOUSEHOLDS
        rng = self.np_random

        # True state of each household (what is really happening)
        self.food = rng.uniform(0.20, 0.50, n)
        self.income = rng.uniform(0.05, 0.30, n)
        self.savings = rng.uniform(0.00, 0.10, n)
        self.health = rng.uniform(0.40, 0.80, n)
        self.school = rng.uniform(0.30, 0.70, n)
        self.dependency = np.zeros(n)

        # What the program has already given each household
        self.has_asset = np.zeros(n, dtype=bool)
        self.has_training = np.zeros(n, dtype=bool)
        self.in_savings_group = np.zeros(n, dtype=bool)
        self.has_health_cover = np.zeros(n, dtype=bool)
        self.asset_age = np.zeros(n)

        # Graduation tracking
        self.graduated = np.zeros(n, dtype=bool)
        self.weeks_since_graduation = np.zeros(n)
        self.graduation_checked = np.zeros(n, dtype=bool)
        self.graduation_held = np.zeros(n, dtype=bool)

        # The officer's records. They start from a baseline survey and only
        # get updated when staff visit the household.
        self.record_food = self.food.copy()
        self.record_income = self.income.copy()
        self.record_savings = self.savings.copy()
        self.record_health = self.health.copy()
        self.record_school = self.school.copy()
        self.record_dependency = self.dependency.copy()
        self.weeks_since_visit = np.zeros(n)

        # Program level state
        self.week = 0
        self.budget = START_BUDGET
        self.drought_weeks_left = 0
        self.current = 0  # household under review this week

        # Counters for reporting
        self.successful_graduations = 0
        self.relapses = 0
        self.last_action = None
        self.last_events = []

        return self._get_observation(), self._get_info()

    # -----------------------------------------------------------------------
    # Step: one week of the program
    # -----------------------------------------------------------------------
    def step(self, action):
        action = int(action)
        h = self.current
        reward = 0.0
        self.last_events = []

        score_before = self._lasting_score().mean()

        # 1. Apply the chosen action to the household under review
        cost = ACTION_COSTS[action]
        if cost > self.budget:
            # Not enough money: nothing happens and the agent is penalised
            reward -= 1.0
            self.last_events.append("Action refused: not enough budget")
            action_done = MONITOR
        else:
            self.budget -= cost
            action_done = action
            self._apply_action(h, action)
        self.last_action = action_done

        # 2. One week passes and the officer is scored for it
        reward += self._pass_one_week(score_before)

        # 3. Move to the next household to review
        self.current = self._next_household(h)

        # 4. If every household has graduated, nobody is left to review, but
        #    life goes on. We fast-forward week by week until a household
        #    relapses (it comes back into the program), every graduation is
        #    confirmed, or the program time is over.
        while self.current is None and self.week < PROGRAM_WEEKS and not self._all_graduations_held():
            reward += self._pass_one_week(self._lasting_score().mean())
            self.current = self._next_household(h)

        # The episode ends early only when every household graduated and
        # stayed out of poverty for 12 weeks (terminated). Otherwise it ends
        # when the two year program is over (truncated). Running out of money
        # does not end the program: staff can still monitor and graduate.
        terminated = self._all_graduations_held()
        truncated = bool(self.week >= PROGRAM_WEEKS and not terminated)

        if terminated or truncated:
            reward += self._final_evaluation()
            if self.current is None:
                self.current = h  # keep a valid index for the last observation

        if self.render_mode == "human":
            self.render()

        return self._get_observation(), float(reward), terminated, truncated, self._get_info()

    def _pass_one_week(self, score_before):
        """One week of life for the whole village, and the reward for it."""
        self._simulate_week()

        # Delayed reward: check how graduated households are doing
        reward = self._check_graduations()

        # Progress reward: change in lasting wellbeing across the caseload
        reward += 10.0 * (self._lasting_score().mean() - score_before)

        # Penalty for every household in a food crisis this week
        reward -= 0.3 * (self.food < CRISIS_FOOD_LEVEL).sum()

        self.week += 1
        return reward

    def _all_graduations_held(self):
        # True when every household graduated and passed its 12 week check
        return bool(np.all(self.graduated & self.graduation_held))

    # -----------------------------------------------------------------------
    # What each action does
    # -----------------------------------------------------------------------
    def _apply_action(self, h, action):
        if action == MONITOR:
            return  # nobody visits, records stay as they are

        if action == CASH_TRANSFER:
            # Quick relief: food goes up now, but the household relies on aid more
            self.food[h] += 0.25
            self.dependency[h] += 0.10

        elif action == GIVE_ASSET:
            # Goats, seeds or tools. Pays off slowly after a few weeks.
            if not self.has_asset[h]:
                self.has_asset[h] = True
                self.asset_age[h] = 0

        elif action == SKILLS_TRAINING:
            # Makes the asset more productive
            self.has_training[h] = True

        elif action == HEALTH_SUPPORT:
            # Health cover softens the damage of illness
            self.has_health_cover[h] = True
            self.health[h] += 0.10

        elif action == SAVINGS_GROUP:
            # Village savings group: builds a buffer for hard times
            self.in_savings_group[h] = True

        elif action == ASSESSMENT_VISIT:
            pass  # the only effect is a fresh record (done below)

        elif action == GRADUATE:
            # The household leaves the program and must now stand on its own
            self.graduated[h] = True
            self.weeks_since_graduation[h] = 0
            self.graduation_checked[h] = False
            self.graduation_held[h] = False
            self.last_events.append(f"Household {h + 1} graduated")
            self._clip_values()
            return  # no visit record update when graduating

        self._clip_values()

        # Any staff visit refreshes the officer's record for this household
        self._update_record(h)

    # -----------------------------------------------------------------------
    # One week of life for all households
    # -----------------------------------------------------------------------
    def _simulate_week(self):
        rng = self.np_random
        n = NUM_HOUSEHOLDS

        # Assets only produce income after a delay
        self.asset_age[self.has_asset] += 1
        asset_ready = self.has_asset & (self.asset_age >= ASSET_DELAY_WEEKS)

        # Income growth: needs a working asset, better with training and good
        # health, and weaker when the household depends on handouts
        # Growth also slows down as income gets higher (it cannot grow forever)
        growth = 0.015 * asset_ready * (1.0 + 0.8 * self.has_training)
        growth *= (0.5 + 0.5 * self.health) * (1.0 - 0.5 * self.dependency)
        growth *= (1.0 - self.income)
        self.income += growth - 0.002

        # Food security slowly moves toward what the income can support
        food_target = 0.20 + 0.80 * self.income
        self.food += 0.08 * (food_target - self.food)

        # Savings grow only for members of a savings group
        # (a higher income means more money can be put aside)
        self.savings += 0.012 * (0.2 + self.income) * self.in_savings_group - 0.001

        # Health slowly recovers toward a normal level
        self.health += 0.02 * (0.75 - self.health)

        # Children's school attendance follows food security
        self.school += 0.08 * (self.food - self.school)

        # Dependency fades slowly when no more handouts come
        self.dependency -= 0.004

        # --- Shock 1: drought (hits the whole village for 8 weeks) ---
        if self.drought_weeks_left == 0 and rng.random() < self.drought_chance:
            self.drought_weeks_left = 8
            self.last_events.append("Drought started")

        if self.drought_weeks_left > 0:
            # Savings soften the blow
            protection = 1.0 - 0.7 * self.savings
            self.food -= 0.020 * protection
            self.income -= 0.006 * protection
            self.savings -= 0.010  # families spend savings to cope
            self.drought_weeks_left -= 1

        # --- Shock 2: illness (each household separately) ---
        sick = rng.random(n) < self.illness_chance
        for h in np.where(sick)[0]:
            if self.has_health_cover[h]:
                self.health[h] -= 0.12
                self.income[h] -= 0.02
            else:
                self.health[h] -= 0.30
                self.income[h] -= 0.05
            self.savings[h] -= 0.05
            self.last_events.append(f"Illness in household {h + 1}")

        self._clip_values()

        # Records get older every week
        self.weeks_since_visit += 1
        self.weeks_since_graduation[self.graduated] += 1

    # -----------------------------------------------------------------------
    # Graduation outcomes (this is where the delayed reward comes from)
    # -----------------------------------------------------------------------
    def _check_graduations(self):
        reward = 0.0
        resilience = self._resilience()

        for h in range(NUM_HOUSEHOLDS):
            if not self.graduated[h]:
                continue

            # Relapse: a graduated household falls back into crisis
            if self.food[h] < CRISIS_FOOD_LEVEL:
                reward -= 5.0
                self.relapses += 1
                self.graduated[h] = False  # it comes back into the program
                self.graduation_held[h] = False
                self.last_events.append(f"Household {h + 1} relapsed into crisis")
                continue

            # Check 12 weeks after leaving the program
            if not self.graduation_checked[h] and self.weeks_since_graduation[h] >= GRADUATION_CHECK_WEEKS:
                self.graduation_checked[h] = True
                if self.food[h] >= 0.40 and resilience[h] >= 0.45:
                    # Success: the household stands on its own
                    reward += 5.0
                    self.successful_graduations += 1
                    self.graduation_held[h] = True
                    self.last_events.append(f"Household {h + 1} graduation confirmed")
                else:
                    # Not in crisis, but not strong enough either:
                    # it was graduated too early and comes back into the program
                    reward -= 2.0
                    self.graduated[h] = False
                    self.last_events.append(f"Household {h + 1} not ready, back in program")

        return reward

    def _final_evaluation(self):
        # At the end of the program, reward every household that is on track
        # to stay out of poverty, whether it graduated or not
        on_track = (self.food >= 0.40) & (self._resilience() >= 0.45)
        return 3.0 * on_track.sum()

    # -----------------------------------------------------------------------
    # Scores
    # -----------------------------------------------------------------------
    def _resilience(self):
        # Ability to cope with future shocks without outside help
        base = 0.4 * self.income + 0.3 * self.savings + 0.3 * self.health
        return base * (1.0 - 0.5 * self.dependency)

    def _lasting_score(self):
        # Mix of today's needs (food) and tomorrow's strength (resilience)
        return 0.4 * self.food + 0.6 * self._resilience()

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------
    def _clip_values(self):
        for values in [self.food, self.income, self.savings,
                       self.health, self.school, self.dependency]:
            np.clip(values, 0.0, 1.0, out=values)

    def _update_record(self, h):
        self.record_food[h] = self.food[h]
        self.record_income[h] = self.income[h]
        self.record_savings[h] = self.savings[h]
        self.record_health[h] = self.health[h]
        self.record_school[h] = self.school[h]
        self.record_dependency[h] = self.dependency[h]
        self.weeks_since_visit[h] = 0

    def _next_household(self, h):
        # Review households in a fixed rotation, skipping graduated ones
        for step in range(1, NUM_HOUSEHOLDS + 1):
            candidate = (h + step) % NUM_HOUSEHOLDS
            if not self.graduated[candidate]:
                return candidate
        return None  # everyone has graduated

    def _get_observation(self):
        h = self.current
        obs = np.array([
            self.record_food[h],
            self.record_income[h],
            self.record_savings[h],
            self.record_health[h],
            self.record_school[h],
            self.record_dependency[h],
            float(self.has_asset[h]),
            float(self.has_training[h]),
            float(self.in_savings_group[h]),
            float(self.has_health_cover[h]),
            min(self.weeks_since_visit[h] / 26.0, 1.0),
            min(self.asset_age[h] / 26.0, 1.0),
            self.budget / START_BUDGET,
            1.0 - self.week / PROGRAM_WEEKS,
            float(self.drought_weeks_left > 0),
            self.graduated.mean(),
        ], dtype=np.float32)
        return np.clip(obs, 0.0, 1.0)

    def _get_info(self):
        return {
            "week": self.week,
            "household": self.current,
            "budget": round(float(self.budget), 2),
            "last_action": None if self.last_action is None else ACTION_NAMES[self.last_action],
            "mean_lasting_score": float(self._lasting_score().mean()),
            "households_in_crisis": int((self.food < CRISIS_FOOD_LEVEL).sum()),
            "graduated_now": int(self.graduated.sum()),
            "successful_graduations": self.successful_graduations,
            "relapses": self.relapses,
            "events": list(self.last_events),
        }

    # -----------------------------------------------------------------------
    # Rendering
    # -----------------------------------------------------------------------
    def render(self):
        # "ansi" returns a text table, "human" prints it.
        # The 3D view is served by environment/rendering.py (three.js in the browser).
        if self.render_mode == "ansi":
            return self._text_summary()
        if self.render_mode == "human":
            print(self._text_summary())
        return None

    def _text_summary(self):
        action = "-" if self.last_action is None else ACTION_NAMES[self.last_action]
        lines = [
            f"Week {self.week:3d} | Budget {self.budget:6.1f} | Last action: {action}",
            "  #  food income savings health school dep  status",
        ]
        for h in range(NUM_HOUSEHOLDS):
            status = "graduated" if self.graduated[h] else "in program"
            if self.food[h] < CRISIS_FOOD_LEVEL:
                status += " (CRISIS)"
            lines.append(
                f" {h + 1:2d}  {self.food[h]:.2f}  {self.income[h]:.2f}   {self.savings[h]:.2f}"
                f"   {self.health[h]:.2f}   {self.school[h]:.2f}  {self.dependency[h]:.2f} {status}"
            )
        return "\n".join(lines)

    def close(self):
        pass
