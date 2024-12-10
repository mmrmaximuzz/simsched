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

"""Module for running simulations."""

import contextlib
import itertools
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import TypeAlias

from simsched.engine import (
    SimDeadlock,
    SimOk,
    SimPanic,
    SimThreadConstructor,
    SimTimeout,
    run,
)


@dataclass
class RunStats:
    """Simulation statistics."""

    ok: int = 0
    deadlock: int = 0
    timeout: int = 0
    panic: int = 0


LoopController: TypeAlias = Iterator[None]
LoopControllerConstructor: TypeAlias = Callable[[RunStats], LoopController]


def simsched(
    icoros: Iterable[SimThreadConstructor],
    loopctr: LoopControllerConstructor,
) -> RunStats:
    """Start simulated schedulung."""
    # create a stats object to collect data and communicate with the controller
    stats = RunStats()

    # convert provided iterable to a list to fix the order of coroutines
    coros = list(icoros)

    # run until loopctr returns or CTRL+C
    with contextlib.suppress(KeyboardInterrupt):
        for _ in loopctr(stats):
            match run(coros):
                case SimOk():
                    stats.ok += 1
                case SimDeadlock():
                    stats.deadlock += 1
                case SimTimeout():
                    stats.timeout += 1
                case SimPanic(_):
                    stats.panic += 1

    return stats


# some collection of pre-defined loopers


def time_report_looper(
    reporter: Callable[[float, RunStats], None],
    every_sec: float,
    stats: RunStats,
) -> LoopController:
    """Loop and call `report` every interval provided."""
    start = time.time()
    for _ in itertools.count():
        yield  # run the simulator

        # check how much time passed and report
        end = time.time()
        if end - start >= every_sec:
            reporter(end - start, stats)
            start = end
