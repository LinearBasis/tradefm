"""Numpy-backed circular replay buffer.

Stores (obs, action, reward, next_obs, done) tuples. Actions can be float
(continuous, shape (action_dim,)) or int (discrete index).

Memory layout: pre-allocated arrays of fixed `capacity`, advancing `idx`
modulo capacity. Sampling uses np.random.default_rng for reproducibility.
"""

from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        action_dtype: type = np.float32,  # np.float32 for continuous, np.int64 for discrete
        seed: int = 0,
    ):
        self.capacity = int(capacity)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_dtype = action_dtype

        self._obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        if action_dtype == np.int64:
            self._actions = np.zeros((capacity,), dtype=np.int64)
        else:
            self._actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self._rewards = np.zeros((capacity,), dtype=np.float32)
        self._dones = np.zeros((capacity,), dtype=np.float32)

        self._idx = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    def add(self, obs, action, reward: float, next_obs, done: bool) -> None:
        i = self._idx
        self._obs[i] = obs
        self._next_obs[i] = next_obs
        if self.action_dtype == np.int64:
            self._actions[i] = int(action)
        else:
            self._actions[i] = np.asarray(action, dtype=np.float32)
        self._rewards[i] = reward
        self._dones[i] = float(done)
        self._idx = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Return a dict of np arrays. Caller converts to torch tensors."""
        if self._size == 0:
            raise RuntimeError("ReplayBuffer is empty")
        idx = self._rng.integers(0, self._size, size=batch_size)
        return {
            "obs": self._obs[idx],
            "actions": self._actions[idx],
            "rewards": self._rewards[idx],
            "next_obs": self._next_obs[idx],
            "dones": self._dones[idx],
        }
