"""Topological pattern detection on the plaquette event stream (Workstream B).

We look for ordered traversals of the 4-cycle in the time-ordered event stream
(marks in {A,B,C,D} = {0,1,2,3}):

  * cycle       : A -> B -> C -> D  (repeating)
  * anti-cycle  : A -> D -> C -> B  (repeating)   -- opposite orientation
  * twisted     : A -> B -> C -> D -> A -> D'      -- a cycle then an immediate
                  back-step to D (a defect / recurrence after one traversal)

Detection is by a finite-state matcher walking the stream in time order.  A chain
advances when the next expected vertex fires within the sliding window W of the
previous matched event; it breaks on a wrong vertex or a gap > W.  Each time the
4-step pattern completes we count one cycle and continue expecting the pattern to
repeat, so a single uninterrupted chain may contain many consecutive cycles.

This yields, per chain:
  * its time span (start of first step to last completed step),
  * the number of full cycles it contained.
The stability metrics in `stability_metrics.py` are computed from these chains.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# pattern templates as vertex sequences (length 4)
PATTERNS = {
    "cycle": [0, 1, 2, 3],       # A B C D
    "anti_cycle": [0, 3, 2, 1],  # A D C B
}


@dataclass
class Chain:
    t_start: float
    t_end: float
    n_cycles: int      # number of completed 4-step traversals
    pattern: str

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


def detect_chains(times: np.ndarray, marks: np.ndarray, pattern: list[int],
                  window: float, pattern_name: str = "cycle") -> list[Chain]:
    """Detect maximal chains of the repeating `pattern` in the event stream.

    A chain is a maximal run of events each matching the next expected symbol of
    the (cyclically repeating) pattern, with consecutive matched events at most
    `window` apart in time.  Only chains that complete >= 1 full cycle are
    returned.
    """
    P = pattern
    L = len(P)
    chains: list[Chain] = []

    pos = 0                 # index into pattern of the *next expected* symbol
    chain_start = None      # time of the first matched symbol of the current chain
    last_t = None           # time of the last matched symbol
    n_cycles = 0            # completed cycles in the current chain
    last_complete_t = None  # time at which the last full cycle completed

    def close_chain():
        nonlocal pos, chain_start, last_t, n_cycles, last_complete_t
        if n_cycles >= 1 and chain_start is not None:
            chains.append(Chain(chain_start, last_complete_t, n_cycles, pattern_name))
        pos = 0
        chain_start = None
        last_t = None
        n_cycles = 0
        last_complete_t = None

    for t, m in zip(times, marks):
        if pos == 0:
            # Expecting the pattern's first symbol P[0].  Two sub-cases:
            #  (a) mid-chain (chain_start is not None): the previous cycle just
            #      completed; the chain only survives if P[0] arrives within
            #      `window` of the last matched symbol, otherwise it breaks.
            #  (b) idle (chain_start is None): a P[0] simply starts a new chain.
            if chain_start is not None and last_t is not None and t - last_t > window:
                close_chain()  # persistence broken between cycles
            if m == P[0]:
                if chain_start is None:
                    chain_start = t
                pos = 1
                last_t = t
            # else: non-A event while idle/just-broken -> keep waiting
            continue

        # mid-pattern: need symbol P[pos] within window of last_t
        if t - last_t > window:
            # gap too large -> chain breaks
            close_chain()
            # this event might itself start a new chain
            if m == P[0]:
                chain_start = t
                pos = 1
                last_t = t
            continue

        if m == P[pos]:
            pos += 1
            last_t = t
            if pos == L:
                # completed a full cycle
                n_cycles += 1
                last_complete_t = t
                pos = 0  # expect the pattern to repeat (next symbol P[0])
                # NOTE: chain continues; next P[0] within window keeps it alive
        else:
            # wrong symbol -> chain breaks
            close_chain()
            if m == P[0]:
                chain_start = t
                pos = 1
                last_t = t

    close_chain()
    return chains


def detect_twisted(times: np.ndarray, marks: np.ndarray, window: float) -> int:
    """Count 'twisted' defects: a completed cycle A->B->C->D immediately followed
    (within `window`) by A then a back-step to D, i.e. the sequence
    A B C D A D within consecutive matched symbols.  Returns the count."""
    target = [0, 1, 2, 3, 0, 3]
    count = 0
    pos = 0
    last_t = None
    for t, m in zip(times, marks):
        if pos > 0 and last_t is not None and t - last_t > window:
            pos = 1 if m == target[0] else 0
            last_t = t if pos == 1 else None
            continue
        if m == target[pos]:
            pos += 1
            last_t = t
            if pos == len(target):
                count += 1
                pos = 0
                last_t = None
        else:
            pos = 1 if m == target[0] else 0
            last_t = t if pos == 1 else None
    return count
