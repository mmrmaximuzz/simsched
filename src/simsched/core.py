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
    FINAL = auto()


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


def finish() -> SimThread:
    """Finish the current simulation thread.

    The scheduler will interpret this thread as finished on next poll cycle. It
    is used to avoid diddling with StopIteration, but it is also useful to
    completely stop the thread from some deeply-nested coroutine without
    passing dozens of return codes throughout.
    """
    cmd = yield ThreadState.YIELD
    while True:
        if cmd != SchedulerMessage.POLL:
            raise AssertionError(f"bad command for finished thread: {cmd}")

        cmd = yield ThreadState.FINAL
