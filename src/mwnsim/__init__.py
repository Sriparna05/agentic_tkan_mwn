"""
MWNSim: Mobile Wireless Network Simulation Package for O-RAN Slicing & Telemetry
"""

from .environment import WirelessSlicingEnv

# Backward-compatible alias
MobileWirelessEnv = WirelessSlicingEnv

__all__ = ["WirelessSlicingEnv", "MobileWirelessEnv"]
