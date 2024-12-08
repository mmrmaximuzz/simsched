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
from simsched.engine import run


def test_single_thread():
    """The single thread must spawn and complete when being run."""
    steps = [False, False]

    def thread() -> SimThread:
        """Simulate some thread interleaving."""
        steps[0] = True
        yield from schedule()
        steps[1] = True

    assert run([thread]), "must not deadlock"
    assert all(steps), "must be executed till the end"


def test_explicit_finish():
    """The single thread must stop when finish is called."""
    steps = [False, False]

    def thread() -> SimThread:
        """Simulate thread early exit."""
        steps[0] = True
        yield from finish()
        steps[1] = True  # unreachable

    assert run([thread]), "must not deadlock"
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

    assert run([thread]), "must not deadlock"
    assert steps == [True, False], "must not execute after finish"


def test_multiple_threads():
    """All the threads must spawn and complete."""
    nr_threads = 5
    executed = [False] * nr_threads

    def thread(n: int) -> SimThread:
        yield from schedule()
        executed[n] = True

    assert run(partial(thread, i) for i in range(nr_threads)), "no deadlock"
    assert all(executed), "all executed"


def test_simple_deadlock():
    """Engine must detect deadlock when some threads cannot finish."""
    steps = [False, False]

    def thread() -> SimThread:
        """Simulate explicit deadlock."""
        steps[0] = True
        yield from cond_schedule(lambda: False)
        steps[1] = True  # unreachable

    assert not run([thread]), "must detect deadlock"
    assert steps == [True, False], "must stop when deadlocked"
