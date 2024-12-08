"""Engine module for simsched tool."""

import contextlib
import random
from collections.abc import Callable, Iterable
from typing import TypeAlias

from .core import SchedulerMessage, SimThread, ThreadState, schedule

SimThreadConstructor: TypeAlias = Callable[[], SimThread]


def poll(threads: list[SimThread]) -> tuple[list[SimThread], list[SimThread]]:
    """Poll active threads.

    Filters out thread that are already finished. The rest are polled and
    classified and the result is returned - the first tuple item is runnable
    threads, and the second one is all the unfinished threads.
    """

    runnables = []
    available = []

    for t in threads:
        with contextlib.suppress(StopIteration):
            match t.send(SchedulerMessage.POLL):
                case ThreadState.READY:
                    runnables.append(t)

            # if not raised StopIteration, then the thread is not finished
            available.append(t)

    return runnables, available


def run(coros: Iterable[SimThreadConstructor]):
    """Run the simulation engine till the end of execution.

    This is the main function of the simthread tool. It spawns all the
    coroutines provided and simulates it random interleaving until all the
    threads are finished or a deadlock is discovered.
    """
    threads = [coro() for coro in coros]

    # spawn all the generators
    for t in threads:
        next(t)

    while True:
        runnables, available = poll(threads)
        if not runnables:
            if not available:
                print("done - all threads are finished")
            else:
                print("failed - deadlock detected")
            break

        t = random.choice(runnables)
        with contextlib.suppress(StopIteration):
            t.send(SchedulerMessage.CONT)


def test_core():
    trace = []

    def thrd1() -> SimThread:
        yield from schedule()
        trace.append("A")
        yield from schedule()
        trace.append("B")

    def thrd2() -> SimThread:
        yield from schedule()
        trace.append("X")
        yield from schedule()
        trace.append("Y")

    run((thrd1, thrd2))

    print(trace)
