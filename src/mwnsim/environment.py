"""
Mobile Wireless Network Gym-like Simulation Environment.
Models base stations, mobile user equipments (UEs), channel conditions,
interference, path loss, and resource block allocation.
"""

from typing import Dict, Any, Tuple
import numpy as np


class MobileWirelessEnv:
    """
    Simulates a cellular / mobile wireless network with multi-carrier resource allocation.
    """

    def __init__(
        self,
        n_base_stations: int = 3,
        n_users: int = 10,
        max_steps: int = 100,
        bandwidth_mhz: float = 20.0,
        seed: int = 42,
    ):
        self.n_base_stations = n_base_stations
        self.n_users = n_users
        self.max_steps = max_steps
        self.bandwidth_mhz = bandwidth_mhz
        self.current_step = 0
        self.rng = np.random.default_rng(seed)

        # BS locations placed linearly or in a grid
        self.bs_positions = np.linspace(100, 900, self.n_base_stations)
        self.user_positions = np.zeros(self.n_users)
        self.user_velocities = np.zeros(self.n_users)
        self.channel_gains = np.zeros((self.n_base_stations, self.n_users))
        self.user_demands = np.zeros(self.n_users)

        self.reset()

    def reset(self) -> Dict[str, np.ndarray]:
        """Resets the environment state."""
        self.current_step = 0
        # Initialize user positions uniformly in 0..1000m
        self.user_positions = self.rng.uniform(50, 950, self.n_users)
        # Random velocity between -1.5 and 1.5 m/s (pedestrian mobility)
        self.user_velocities = self.rng.uniform(-1.5, 1.5, self.n_users)
        self.user_demands = self.rng.exponential(2.0, self.n_users) + 0.5
        self._update_channel_gains()

        return self._get_obs()

    def _update_channel_gains(self):
        """Computes path loss and small-scale Rayleigh fading."""
        for b in range(self.n_base_stations):
            dist = np.abs(self.user_positions - self.bs_positions[b]) + 10.0  # avoid 0 dist
            # Simple path loss: 128.1 + 37.6 * log10(d_km)
            d_km = dist / 1000.0
            path_loss_db = 128.1 + 37.6 * np.log10(np.maximum(d_km, 0.01))
            fading = self.rng.rayleigh(scale=1.0, size=self.n_users)
            gain_linear = (10 ** (-path_loss_db / 10.0)) * (fading ** 2)
            self.channel_gains[b] = gain_linear

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Returns the current state observation."""
        return {
            "channel_gains": self.channel_gains.copy(),
            "user_demands": self.user_demands.copy(),
            "user_positions": self.user_positions.copy(),
            "step": np.array([self.current_step], dtype=np.int32),
        }

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """
        Takes an action: allocation of BS to users or resource weights.
        action: array of shape (n_users,), values indicating assigned BS index [0 .. n_bs - 1].
        """
        self.current_step += 1

        # Calculate received SNR and Shannon rate for assigned links
        user_throughput = np.zeros(self.n_users)
        noise_power = 1e-13  # ~ -100 dBm thermal noise
        tx_power = 1.0       # 1 Watt

        for u in range(self.n_users):
            assigned_bs = int(np.clip(action[u], 0, self.n_base_stations - 1))
            sig_power = tx_power * self.channel_gains[assigned_bs, u]
            snr = sig_power / noise_power
            # Shannon capacity: B * log2(1 + SNR) in Mbps
            rate = self.bandwidth_mhz * np.log2(1.0 + np.maximum(snr, 1e-6))
            user_throughput[u] = rate

        total_throughput = float(np.sum(user_throughput))
        # Satisfaction ratio based on demands
        satisfaction = np.minimum(user_throughput / (self.user_demands + 1e-6), 1.0)
        reward = float(np.mean(satisfaction) * 10.0 + (total_throughput / 100.0))

        # Mobility update
        self.user_positions += self.user_velocities
        # Reflect at borders
        self.user_positions = np.clip(self.user_positions, 10.0, 990.0)
        # New stochastic demands and channel gains
        self.user_demands = self.rng.exponential(2.0, self.n_users) + 0.5
        self._update_channel_gains()

        done = self.current_step >= self.max_steps
        info = {
            "total_throughput_mbps": total_throughput,
            "mean_satisfaction": float(np.mean(satisfaction)),
            "user_throughput": user_throughput,
        }

        return self._get_obs(), reward, done, info
