"""Engine module for simsched tool."""

import contextlib
import random

from .core import SimThread, ThreadState, SchedulerMessage, schedule


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


TRACE = []


def thrd1() -> SimThread:
    yield from schedule()
    TRACE.append("A")
    yield from schedule()
    TRACE.append("B")


def thrd2() -> SimThread:
    yield from schedule()
    TRACE.append("X")
    yield from schedule()
    TRACE.append("Y")


def test_core():
    threads = [thrd1(), thrd2()]

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

    print(TRACE)
