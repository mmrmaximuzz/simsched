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

from functools import partial

from simsched.core import SimThread, cond_schedule, finish, schedule
from simsched.engine import SimDeadlock, SimOk, SimPanic, SimTimeout, run


def test_single_thread():
    """The single thread must spawn and complete when being run."""
    steps = [False, False]

    def thread() -> SimThread:
        """Simulate some thread interleaving."""
        steps[0] = True
        yield from schedule()
        steps[1] = True

    assert run([thread]) == SimOk(), "must not deadlock"
    assert all(steps), "must be executed till the end"


def test_explicit_finish():
    """The single thread must stop when finish is called."""
    steps = [False, False]

    def thread() -> SimThread:
        """Simulate thread early exit."""
        steps[0] = True
        yield from finish()
        steps[1] = True  # unreachable

    assert run([thread]) == SimOk(), "must not deadlock"
    assert steps == [True, False], "must not execute after finish"


def test_nested_finish():
    """The thread must finished when called from sub-coroutines."""
    steps = [False, False]

    def calls_exit() -> SimThread:
        """Just call `finish`."""
        yield from finish()

    def thread() -> SimThread:
        """Simulate thread early exit."""
        steps[0] = True
        yield from calls_exit()
        steps[1] = True  # unreachable

    assert run([thread]) == SimOk(), "must not deadlock"
    assert steps == [True, False], "must not execute after finish"


def test_multiple_threads():
    """All the threads must spawn and complete."""
    nr_threads = 5
    executed = [False] * nr_threads

    def thread(n: int) -> SimThread:
        yield from schedule()
        executed[n] = True

    threads = (partial(thread, i) for i in range(nr_threads))
    assert run(threads) == SimOk(), "no deadlock"
    assert all(executed), "all executed"


def test_simple_deadlock():
    """Engine must detect deadlock when some threads cannot finish."""
    steps = [False, False]

    def thread() -> SimThread:
        """Simulate explicit deadlock."""
        steps[0] = True
        yield from cond_schedule(lambda: False)
        steps[1] = True  # unreachable

    assert run([thread]) == SimDeadlock(), "must detect deadlock"
    assert steps == [True, False], "must stop when deadlocked"


def test_timeout():
    """Engine must detect timeouts."""

    def thread() -> SimThread:
        """Simulate long thread."""
        # step 1
        yield from schedule()
        # step 2
        yield from schedule()
        # step 3
        yield from schedule()
        # step 4

    assert run([thread], max_steps=3) == SimTimeout()
    assert run([thread], max_steps=4) == SimOk()


def test_loop_detection():
    """Engine must detect loops with default value of max_steps."""

    def thread() -> SimThread:
        """Simulate endless thread."""
        while True:
            yield from schedule()

    assert run([thread]) == SimTimeout(), "endless loops must be timeouted"


def test_simple_exception():
    """Engine must catch exceptions when advancing threads."""

    def thread() -> SimThread:
        """Simulate thread with error happened."""
        yield from schedule()
        raise RuntimeError("testmsg")

    res = run([thread])
    assert isinstance(res, SimPanic), "result type must be PANIC"
    assert isinstance(res.e, RuntimeError), "must provide the exception object"
    assert res.e.args == ("testmsg",), "must provide the exception object"
