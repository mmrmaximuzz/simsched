"""Core components for simsched engine."""

import contextlib
import random
from collections.abc import Callable, Generator
from typing import TypeAlias


TRACE = []

SimThread: TypeAlias = Generator[str, str, None]
Predicate: TypeAlias = Callable[[], bool]


def cond_schedule(is_runnable: Predicate) -> SimThread:
    """Schedule the thread with wakeup condition.

    This is the main scheduling primitive. It returns control to the scheduler
    engine and waits for its commands. When polled, returns the predicate
    result. When got :continue: command, then returns control to the caller
    simthread.
    """
    cmd = yield "seqpoint"
    while True:
        match cmd:
            case "poll":
                cmd = yield "ready" if is_runnable() else "blocked"
            case "continue":
                break
            case _:
                raise AssertionError(f"unknown schedule command {cmd}")


def schedule() -> SimThread:
    """Simply yield the current thread execution to the engine.

    Useful for simulating interleaving code paths.
    """
    yield from cond_schedule(lambda: True)


def thrd1() -> SimThread:
    yield from schedule()
    TRACE.append("T1 1")
    yield from schedule()
    TRACE.append("T1 2")


def thrd2() -> SimThread:
    yield from schedule()
    TRACE.append("T2 1")
    yield from schedule()
    TRACE.append("T2 2")
    while True:
        yield "blocked"


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
            match t.send("poll"):
                case "ready":
                    runnables.append(t)

            # if not raised StopIteration, then the thread is not finished
            available.append(t)

    return runnables, available


def test_core():
    t1 = thrd1()
    t2 = thrd2()
    assert isinstance(t1, Generator)
    assert isinstance(t2, Generator)

    threads = [t1, t2]

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
            t.send("continue")

    print(TRACE)
