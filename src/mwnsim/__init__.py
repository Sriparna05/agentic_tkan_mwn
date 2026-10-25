"""
MWNSim: Mobile Wireless Network Simulation Package for O-RAN Slicing & Telemetry
"""

from .environment import WirelessSlicingEnv
from .policies import (
    BaseControlPolicy,
    EqualPeriodicPolicy,
    QueueAwarePeriodicPolicy,
    QueueAwareAgeTriggeredPolicy,
    QueueAwareUncertaintyPolicy,
    DecisionSensitivePolicy,
    PrivilegedFullInfoPolicy,
    allocate_integer_prbs
)

# Backward-compatible alias
MobileWirelessEnv = WirelessSlicingEnv

__all__ = [
    "WirelessSlicingEnv",
    "MobileWirelessEnv",
    "BaseControlPolicy",
    "EqualPeriodicPolicy",
    "QueueAwarePeriodicPolicy",
    "QueueAwareAgeTriggeredPolicy",
    "QueueAwareUncertaintyPolicy",
    "DecisionSensitivePolicy",
    "PrivilegedFullInfoPolicy",
    "allocate_integer_prbs"
]
