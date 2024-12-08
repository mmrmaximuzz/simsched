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


def run(coros: Iterable[SimThreadConstructor]) -> bool:
    """Run the simulation engine till the end of execution.

    This is the main function of the simthread tool. It spawns all the
    coroutines provided and simulates it random interleaving until all the
    threads are finished or a deadlock is discovered.
    """
    threads = spawn_coroutines(coros)

    while True:
        runnables, available = poll(threads)
        if not runnables:
            if not available:
                # no runnables and all finished - OK
                return True
            else:
                # no runnables but some are not finished - DEADLOCK
                return False

        # pick up the random thread and advance it
        thrd = random.choice(runnables)
        state = thrd.send(SchedulerMessage.CONT)

        # put some asserts to catch errors early
        assert state == ThreadState.YIELD, state
