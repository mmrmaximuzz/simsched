# simsched - simulated scheduling tool for investigating concurrency issues
# Copyright (C) Maxim Petrov 2024
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

"""This is a demo example for the simsched project.

Classic example with mutexes taken in the wrong order in different threads,
which could lead to a deadlock.

This example collects the traces leading to a deadlock situation.
"""

from functools import partial
from typing import TypeAlias

from simsched.core import SimThread
from simsched.engine import SimDeadlock
from simsched.lib import Mutex
from simsched.runner import LoopController, RunStats, simsched

Event: TypeAlias = tuple[int, str]
Trace: TypeAlias = tuple[Event, ...]
TraceBuffer: TypeAlias = list[Event]


def thread(tid: int, x: Mutex, y: Mutex, trace: TraceBuffer) -> SimThread:
    """Lock two mutexes."""
    yield from x.lock()
    trace.append((tid, f"LOCK {x.label}"))
    yield from y.lock()
    trace.append((tid, f"LOCK {y.label}"))

    yield from y.unlock()
    trace.append((tid, f"UNLOCK {y.label}"))
    yield from x.unlock()
    trace.append((tid, f"UNLOCK {x.label}"))


def demo() -> None:
    """Simulate the classic deadlock problem."""
    mtx_a = Mutex(label="A")
    mtx_b = Mutex(label="B")
    mtx_c = Mutex(label="C")

    tracebuf: TraceBuffer = []
    deadlocks: set[Trace] = set()

    def looper(stats: RunStats) -> LoopController:
        """Collect deadlock states."""
        yield  # first run
        while True:
            if stats.last == SimDeadlock():
                trace = tuple(tracebuf)
                new = trace not in deadlocks
                deadlocks.add(trace)
                if new:
                    print(f"-> deadlock discovered ({len(deadlocks)} total)")

            # clear the state before the next run
            mtx_a.locked = False
            mtx_b.locked = False
            mtx_c.locked = False
            tracebuf.clear()
            yield

    print("Demo - mutexes")
    print("Classic deadlock with 3 threads")
    print("Hit Ctrl+C to stop simulation")
    print("Running...")

    t0 = partial(thread, 0, mtx_a, mtx_b, tracebuf)
    t1 = partial(thread, 1, mtx_b, mtx_c, tracebuf)
    t2 = partial(thread, 2, mtx_c, mtx_a, tracebuf)
    stats = simsched((t0, t1, t2), looper)

    print()
    print(f"total({stats.total}) = ok({stats.ok}) + lock({stats.deadlock})")
    print("observed the following deadlocks:")
    for i, deadlock in enumerate(sorted(deadlocks), 1):
        print(f"========================== #{i} ==========================")
        print("T0", " " * 16, "T1", " " * 16, "T2")
        print("--------------------------------------------------------")

        for tid, line in deadlock:
            print(" " * 20 * tid + line)

        print(">>>>>>>>>>>>>>>>>>> D E A D L O C K <<<<<<<<<<<<<<<<<<<")
        print()


def main() -> None:
    """CLI entrypoint."""
    demo()


if __name__ == "__main__":
    main()
