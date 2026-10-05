import numpy as np


MAX_ACTION = 0.05

OBSTACLE = {
    "xmin": -0.20,
    "xmax": 0.20,
    "ymin": 0.35,
    "ymax": 0.65,
}

LEFT_WAYPOINT = np.array([-0.35, 0.50])
RIGHT_WAYPOINT = np.array([0.35, 0.50])


def step(position, action):
    norm = np.linalg.norm(action)

    if norm > MAX_ACTION:
        action = action / norm * MAX_ACTION

    return position + action


def make_observation(position, goal):
    return np.concatenate([position, goal])


def move_towards(position, target):
    direction = target - position
    distance = np.linalg.norm(direction)

    if distance < 1e-8:
        return np.zeros(2)

    return direction / distance * min(distance, MAX_ACTION)


def waypoint_expert(position, goal, waypoint):
    if position[1] < waypoint[1]:
        target = waypoint
    else:
        target = goal

    return move_towards(position, target)


def left_expert(position, goal):
    return waypoint_expert(position, goal, LEFT_WAYPOINT)


def right_expert(position, goal):
    return waypoint_expert(position, goal, RIGHT_WAYPOINT)


def rollout(policy, start, goal, max_steps=100, goal_tolerance=1e-3):
    position = start.copy()

    observations = []
    actions = []
    positions = [position.copy()]

    for _ in range(max_steps):
        observation = make_observation(position, goal)
        action = policy(position, goal)

        observations.append(observation)
        actions.append(action)

        position = step(position, action)
        positions.append(position.copy())

        if np.linalg.norm(position - goal) < goal_tolerance:
            break

    return {
        "observations": np.asarray(observations),
        "actions": np.asarray(actions),
        "positions": np.asarray(positions),
    }


def sample_task(rng):
    start = np.array([
        rng.uniform(-0.1, 0.1),
        0.0,
    ])

    goal = np.array([
        rng.uniform(-0.1, 0.1),
        1.0,
    ])

    return start, goal


def collect_demonstrations(n_episodes, seed=42):
    rng = np.random.default_rng(seed)

    demonstrations = []

    for _ in range(n_episodes):
        start, goal = sample_task(rng)

        if rng.random() < 0.5:
            expert = left_expert
            strategy = "left"
        else:
            expert = right_expert
            strategy = "right"

        trajectory = rollout(
            expert,
            start,
            goal,
        )

        trajectory["strategy"] = strategy
        demonstrations.append(trajectory)

    return demonstrations


# Task-level metrics
def is_collision(position):
    x, y = position

    return (
        OBSTACLE["xmin"] <= x <= OBSTACLE["xmax"]
        and OBSTACLE["ymin"] <= y <= OBSTACLE["ymax"]
    )


def reached_goal(position, goal, tolerance=0.05):
    return np.linalg.norm(position - goal) <= tolerance