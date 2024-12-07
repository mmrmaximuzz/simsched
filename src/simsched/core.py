"""Core components for simsched engine."""

from collections.abc import Callable, Generator
from enum import Enum, auto
from typing import TypeAlias


class ThreadState(Enum):
    """The thread's state of execution.

    Events of this type are sent from threads to the simulation engine.
    """

    YIELD = auto()
    READY = auto()
    BLOCK = auto()


class SchedulerMessage(Enum):
    """Some command from the engine.

    Events of this type are messages from the simulation engine for threads.
    """

    POLL = auto()
    CONT = auto()


# some type definitions
SimThread: TypeAlias = Generator[ThreadState, SchedulerMessage, None]
Predicate: TypeAlias = Callable[[], bool]


def cond_schedule(is_runnable: Predicate) -> SimThread:
    """Schedule the thread with wakeup condition.

    This is the main scheduling primitive. It returns control to the scheduler
    engine and waits for its commands. When `poll`ed, returns the predicate
    result. When got `continue` command, then returns control to the caller
    simthread.
    """
    cmd = yield ThreadState.YIELD
    while True:
        match cmd:
            case SchedulerMessage.POLL:
                if is_runnable():
                    cmd = yield ThreadState.READY
                else:
                    cmd = yield ThreadState.BLOCK
                continue
            case SchedulerMessage.CONT:
                break


def schedule() -> SimThread:
    """Simply yield the current thread execution to the engine.

    Useful for simulating interleaving code paths.
    """
    yield from cond_schedule(lambda: True)
