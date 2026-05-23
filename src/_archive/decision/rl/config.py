"""Config dataclasses for RL training.

Convention follows src/config.py: flat dataclasses, JSON-loadable via
src.config.load_from_json. Three configs separated by concern:

    EnvConfig   — what the environment does (action mode, reward, latency, A-S baseline)
    AgentConfig — RL algorithm hyperparams (per-algorithm subset is used)
    TrainConfig — how training is orchestrated (rollouts, seed, logging)

A run config file may contain all three nested under "env"/"agent"/"train" keys,
or three separate JSONs may be passed to the launcher.
"""

from dataclasses import dataclass
from typing import Literal


# Action mode keys used throughout the codebase.
#   a       — continuous residual over A-S, (Δhalf_spread, Δskew)
#   b       — continuous A-S param tuning, (γ-mult, skew-mult)
#   c       — continuous direct β·spread, (β_bid, β_ask), no A-S
#   a_disc  — 9 actions, residual-A-S grid
#   b_disc  — 20 actions, Alpha-AS grid (γ × skew)
#   c_disc  — 10 actions, Smirnov grid (ask×bid book levels)
ActionMode = Literal["a", "b", "c", "a_disc", "b_disc", "c_disc"]

# State builder mode.
#   "heads"  — α/σ/κ + market features (~14 dim)
#   "hidden" — h_t (d_model) + market features
#   "both"   — α/σ/κ + h_t + market features (3 + d_model + 11 dim)
StateMode = Literal["heads", "hidden", "both"]

# Algorithm tag. Discrete-only modes (*_disc) require D3QN; continuous modes
# require one of {sac, ddpg, td3}.
Algorithm = Literal["sac", "ddpg", "td3", "d3qn"]


@dataclass
class EnvConfig:
    """Everything that defines a single rollout episode."""

    # --- What to backtest ---
    instrument: str = "SBER"
    date: str = "2024-03-22"
    data_dir: str = "data/hftbacktest"

    # --- Tokens (mirrors heads-training pipeline) ---
    # Manifest gives a stable inst_id (alphabetical position in
    # manifest["instruments"]) — same convention the transformer's instrument
    # embedding was trained with. By default token data comes from
    # sequences/<INST>_<DATE>.parquet which must contain `time_sec`
    # (pipeline.py:87 includes it). For ad-hoc backtests, override with
    # an explicit file (matches backtest_orig_as.py style).
    sequences_dir: str = "data/processed/sequences"
    tokens_parquet_override: str | None = None

    # --- Frozen model checkpoints (used for state/A-S inputs) ---
    transformer_checkpoint: str = "checkpoints/best.pt"
    heads_checkpoint: str = "checkpoints/heads/best_heads.pt"

    # --- What kind of state / action / reward ---
    action_mode: ActionMode = "a"
    state_mode: StateMode = "heads"

    # --- Action parameterization constants (Modes A/B; Mode C uses book spread directly) ---
    # Mode A: half_spread_final = half_spread_AS * exp(c_spread * a_spread)
    #          skew_final        = skew_AS_inv + c_skew * a_skew * σ̂ * Δt
    c_spread: float = 0.7  # ±exp(0.7) ≈ ×0.5..×2.0
    c_skew: float = 1.0    # in units of σ·Δt
    # Mode B: γ_final = γ_AS * exp(c_gamma * a_gamma);  prices *= (1 + c_skew_b * a_skew)
    c_gamma: float = 1.0
    c_skew_b: float = 0.1

    # --- A-S baseline params (driving Modes A/B) ---
    gamma_as: float = 0.1
    sigma_floor: float = 1e-5  # clip σ̂ from head to ≥ this to avoid zero spread

    # --- Cap to keep policy from drifting into no-quote regime ---
    max_half_spread_ticks: float = 20.0

    # --- Reward (quadratic A-S-CARA-style running utility) ---
    # r_t = scale * [ ΔPnL_MTM_t − (γ_R/2) · q_norm_t² · σ̂_t² · Δt ]
    gamma_R: float = 0.1
    reward_scale: float = 2.0  # ≈ 1/(tick * max_inventory) for SBER: 1/(0.01*50) = 2.0

    # --- Episode timing ---
    quote_refresh_ms: float = 1000.0  # Δt for both action cadence and reward
    flatten_min_before_end: float = 5.0
    # Truncate episode after this many steps. None → run whole session.
    # Useful for fast smoke runs / debugging.
    max_steps_per_rollout: int | None = None

    # --- Inventory / sizes ---
    order_qty: float = 1.0
    max_inventory: int = 50

    # --- Venue ---
    tick_size: float = 0.01
    lot_size: float = 1.0
    latency_ms: float = 30.0
    maker_fee: float = -0.00005
    taker_fee: float = 0.00050

    # --- Misc ---
    device: str = "cpu"  # device for the frozen transformer/heads at inference


@dataclass
class AgentConfig:
    """Hyperparams for the RL algorithm. Per-algorithm subset is read."""

    algorithm: Algorithm = "sac"

    # --- Common ---
    discount: float = 0.99   # γ_RL per step (≡ per second at 1Hz refresh)
    batch_size: int = 256
    buffer_size: int = 1_000_000
    learn_starts: int = 5_000  # collect this many env steps before any SGD
    target_update_tau: float = 0.005  # soft target update τ

    # MLP architecture (actor and critic share)
    hidden_dim: int = 256
    n_hidden_layers: int = 2

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    # --- SAC ---
    sac_auto_alpha: bool = True
    sac_init_log_alpha: float = 0.0
    sac_target_entropy: float | None = None  # None → -action_dim

    # --- DDPG / TD3 ---
    action_noise_std: float = 0.1
    td3_policy_delay: int = 2
    td3_target_noise: float = 0.2
    td3_target_noise_clip: float = 0.5

    # --- D3QN ---
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000
    d3qn_target_update_freq: int = 1000  # hard target update every N grad steps
    d3qn_lr: float = 1e-4


@dataclass
class TrainConfig:
    """Training-loop orchestration."""

    seed: int = 0
    n_rollouts: int = 30  # one rollout = full session (≈31_500 transitions @ 1Hz)
    updates_per_step: int = 1  # SGD updates per env step (for off-policy algos)

    # Logging / checkpointing
    log_dir: str = "runs/rl"
    run_name: str | None = None  # autogen if None
    log_every_steps: int = 200
    eval_every_rollouts: int = 5
    save_every_rollouts: int = 10
