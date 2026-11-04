"""Configuration for a 10BASE-T1S PLCA multidrop segment.

All internal times are expressed in *bit times* (BT). At 10 Mb/s one bit
time is 100 ns, so 10 BT = 1 microsecond.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PLCAConfig:
    # Number of nodes on the segment, including the coordinator (node 0).
    node_count: int = 8
    # Physical bit rate of 10BASE-T1S.
    bit_rate: float = 10e6
    # Length of an empty transmit opportunity (IEEE 802.3cg default: 32 BT).
    to_timer_bt: int = 32
    # Duration of the BEACON sent by the coordinator (20 BT).
    beacon_bt: int = 20
    # PLCA burst mode: extra frames a node may send in one TO (0 = off).
    max_bc: int = 0
    # Per-node, per-class transmit queue limit in frames; extra arrivals are
    # dropped. Critical and non-critical frames have separate queues, so a
    # backlog of non-critical frames can never push out a critical frame.
    max_queue: int = 64
    # Inter-packet gap after each frame.
    ipg_bt: int = 96
    # Preamble + start-of-frame delimiter in bytes.
    preamble_bytes: int = 8

    def wire_bits(self, frame_bytes: int) -> int:
        """Bus time (BT) used by one frame of `frame_bytes` (MAC header..FCS)."""
        return (frame_bytes + self.preamble_bytes) * 8 + self.ipg_bt

    def s_to_bt(self, seconds: float) -> float:
        return seconds * self.bit_rate

    def bt_to_us(self, bt: float) -> float:
        return bt / self.bit_rate * 1e6

    @property
    def empty_cycle_bt(self) -> int:
        """Length of a PLCA cycle in which no node transmits."""
        return self.beacon_bt + self.node_count * self.to_timer_bt
