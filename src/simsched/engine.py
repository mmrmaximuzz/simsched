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

"""Engine module for simsched tool."""

import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TypeAlias

from .core import SchedulerMessage, SimThread, ThreadState, finish, schedule

SimThreadConstructor: TypeAlias = Callable[[], SimThread]


def poll(threads: list[SimThread]) -> tuple[list[SimThread], list[SimThread]]:
    """Poll active threads.

    Filters out thread that are already finished. The rest are polled and
    classified and the result is returned - the first tuple item is runnable
    threads, and the second one is all the unfinished threads.
    """
    ready = []
    total = []

    for thrd in threads:
        match thrd.send(SchedulerMessage.POLL):
            case ThreadState.FINAL:
                # stop accounting threads which reported it is finished
                continue
            case ThreadState.READY:
                ready.append(thrd)

        total.append(thrd)

    return ready, total


def spawn_coroutines(ts: Iterable[SimThreadConstructor]) -> list[SimThread]:
    """Prepare the actual coroutine objects from constructors."""

    def simthread_wrapper(t: SimThreadConstructor) -> SimThread:
        """The wrapper simthread.

        This wrapper is a tricky one. It wraps the user-provided simthread
        between early `schedule` call (to not force the user simthreads to
        synchronize the thread start by themselves), and the lately `finish`
        call to avoid StopIteration handling.
        """
        yield from schedule()
        yield from t()
        yield from finish()

    threads = [simthread_wrapper(t) for t in ts]

    # spawn all the generators
    for thrd in threads:
        state = next(thrd)
        # must stop with `yield` state
        assert state == ThreadState.YIELD, state

    return threads


@dataclass
class SimOk:
    """Engine run ended with OK result."""


@dataclass
class SimDeadlock:
    """Engine run ended with deadlock."""


@dataclass
class SimTimeout:
    """Engine did not finish the run due to number of steps being exceeded."""


@dataclass
class SimPanic:
    """Engine catched some exception when advancing some thread."""

    e: Exception


SimResult: TypeAlias = SimOk | SimDeadlock | SimTimeout | SimPanic


def run(
    coros: Iterable[SimThreadConstructor],
    max_steps: int = 1000,
) -> SimResult:
    """Run the simulation engine till we get some result.

    This is the main function of the simthread tool. It spawns all the
    coroutines provided and simulates it random interleaving. It stops when
    some condition happened:
    * all the threads are complete - OK
    * on some step only blocked threads remain - DEADLOCK
    * step limit exceeded - TIMEOUT
    * some thread raised an exception - PANIC
    """
    threads = spawn_coroutines(coros)

    # account the fact we need one additional pseudostep to start a thread
    for _ in range(max_steps + len(threads)):
        runnables, available = poll(threads)
        if not runnables:
            if not available:
                # no runnables and all finished - OK
                return SimOk()
            else:
                # no runnables but some are not finished - DEADLOCK
                return SimDeadlock()

        # pick up the random thread to advance
        thrd = random.choice(runnables)
        try:
            # Catch exceptions only when advancing threads with user-provided
            # code. It is the only place where it is permitted to happen - i.e.
            # no exceptions allowed when polling threads.
            state = thrd.send(SchedulerMessage.CONT)
        except Exception as e:
            return SimPanic(e)

        # put some asserts to catch errors early
        assert state == ThreadState.YIELD, state
    else:
        # the loop is completed, too many steps - TIMEOUT
        return SimTimeout()
