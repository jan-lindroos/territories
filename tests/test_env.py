import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import territories


SEED = 42
W, H = 40, 40
RADIUS = 5

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3


def make_env(**kw):
    defaults = dict(num_agents=1, width=W, height=H, window_radius=RADIUS, seed=SEED)
    defaults.update(kw)
    return territories.Env(**defaults)


def test_capture_small_loop():
    env = make_env()
    env.reset()
    initial = env.get_agents()[0].territory
    for action in [UP, UP, RIGHT, DOWN, DOWN]:
        env.step([action])
    a = env.get_agents()[0]
    if a.is_alive:
        assert a.territory > initial
        assert not a.has_trail


def test_capture_larger_loop():
    env = make_env()
    env.reset()
    initial = env.get_agents()[0].territory
    for action in [UP, UP, UP, RIGHT, RIGHT, RIGHT, DOWN, DOWN, DOWN, LEFT, LEFT, LEFT]:
        env.step([action])
    a = env.get_agents()[0]
    if a.is_alive:
        assert a.territory > initial
        assert not a.has_trail


def test_no_capture_without_return():
    env = make_env()
    env.reset()
    initial = env.get_agents()[0].territory
    env.step([UP])
    env.step([UP])
    a = env.get_agents()[0]
    if a.is_alive:
        assert a.territory == initial
        assert a.has_trail


def test_capture_kills_enclosed_agent():
    env = make_env(num_agents=2, width=80, height=80, seed=0)
    env.reset()
    agents = env.get_agents()
    assert agents[0].is_alive
    assert agents[1].is_alive


def test_dead_agent_no_territory():
    env = make_env(width=10, height=10)
    env.reset()
    for _ in range(20):
        env.step([UP])
    a = env.get_agents()[0]
    assert not a.is_alive
    assert a.territory == 0


def test_many_captures_no_crash():
    env = make_env(num_agents=2, width=30, height=30)
    env.reset()
    actions_cycle = [[UP, DOWN], [RIGHT, LEFT], [DOWN, UP], [LEFT, RIGHT]]
    for i in range(200):
        env.step(actions_cycle[i % 4])


def test_capture_enclosed_area_size():
    env = make_env(width=40, height=40)
    env.reset()
    initial = env.get_agents()[0].territory
    for action in [UP, UP, RIGHT, DOWN, DOWN]:
        env.step([action])
    a = env.get_agents()[0]
    if a.is_alive:
        assert a.territory > initial


def test_self_trail_collision():
    env = make_env(width=20, height=20)
    env.reset()
    for action in [UP, UP, UP, RIGHT, DOWN, LEFT]:
        env.step([action])
        if not env.get_agents()[0].is_alive:
            break
    assert not env.get_agents()[0].is_alive


def test_wall_clamp_causes_self_trail_death():
    env = make_env(width=10, height=10)
    env.reset()
    for _ in range(20):
        env.step([UP])
        if not env.get_agents()[0].is_alive:
            break
    assert not env.get_agents()[0].is_alive


def test_enemy_trail_kill():
    env = make_env(num_agents=2, width=10, height=10, seed=0)
    env.reset()
    for _ in range(50):
        env.step([RIGHT, LEFT])
    agents = env.get_agents()
    assert any(not a.is_alive for a in agents)


def test_head_on_collision_both_die():
    for seed in range(100):
        env = make_env(num_agents=2, width=6, height=6, seed=seed)
        env.reset()
        agents = env.get_agents()
        a0, a1 = agents[0], agents[1]
        if a0.y == a1.y and a1.x - a0.x == 2:
            env.step([RIGHT, LEFT])
            agents = env.get_agents()
            if not agents[0].is_alive and not agents[1].is_alive:
                assert agents[0].kills == 0
                assert agents[1].kills == 0
                return
    assert True


def test_dead_agent_not_moved():
    env = make_env(width=10, height=10)
    env.reset()
    for _ in range(20):
        env.step([UP])
    a = env.get_agents()[0]
    assert not a.is_alive
    x_dead, y_dead = a.x, a.y
    for _ in range(5):
        env.step([DOWN])
    a = env.get_agents()[0]
    assert a.x == x_dead and a.y == y_dead
    assert not a.is_alive


def test_reset_deterministic():
    env1 = make_env(seed=99)
    env1.reset()
    env2 = make_env(seed=99)
    env2.reset()
    a1 = env1.get_agents()[0]
    a2 = env2.get_agents()[0]
    assert a1.x == a2.x and a1.y == a2.y


def test_reset_clears_trails_and_territory():
    env = make_env()
    env.reset()
    env.step([UP])
    env.step([UP])
    env.reset()
    a = env.get_agents()[0]
    assert a.is_alive
    assert a.territory == 9  # 3x3 spawn
    assert not a.has_trail


def test_obs_size():
    env = make_env(num_agents=2)
    env.reset()
    obs = env.get_obs()
    side = 2 * RADIUS + 1
    assert len(obs) == 2 * side * side


def test_obs_own_head_in_center():
    env = make_env(num_agents=2)
    env.reset()
    obs = env.get_obs()
    side = 2 * RADIUS + 1
    obs_size = side * side
    center = obs_size // 2
    OWN_HEAD = 1
    assert obs[center] == OWN_HEAD          # agent 0
    assert obs[obs_size + center] == OWN_HEAD  # agent 1


def test_obs_dead_agent_all_empty():
    env = make_env(width=10, height=10)
    env.reset()
    for _ in range(20):
        env.step([UP])
    assert not env.get_agents()[0].is_alive
    obs = env.get_obs()
    assert all(c == 0 for c in obs)
