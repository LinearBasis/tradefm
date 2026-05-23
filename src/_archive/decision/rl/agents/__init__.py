"""RL agents. Each agent is a self-contained class with:

    .select_action(obs, deterministic=False) → np.ndarray | int
    .update(batch) → dict of scalar metrics
    .save(path) / .load(path)

Construction:
    Agent(obs_dim, action_dim_or_n_actions, cfg: AgentConfig, device)
"""
