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

from simsched.core import SimThread, schedule
from simsched.lib import Mutex
from simsched.runner import LoopController, RunStats, simsched


def test_simsched_simple():
    """Runner must be controlled by the looper object."""
    nr_iters = 5

    def looper(stats: RunStats) -> LoopController:
        """Simple looper with upper limit of cycles."""
        for _ in range(nr_iters):
            yield
            # can access the statistics to control execution
            assert stats.ok
            assert stats.deadlock == 0
            assert stats.timeout == 0
            assert stats.panic == 0

    def thread() -> SimThread:
        """Simple thread to execute."""
        yield from schedule()

    stats = simsched([thread], looper)
    assert stats == RunStats(ok=nr_iters, deadlock=0, timeout=0, panic=0)


def test_simsched_abba_deadlock():
    """Runner must catch the simple deadlock."""

    def looper(stats: RunStats) -> LoopController:
        """Loop until deadlock is discovered with some upper limit."""
        # 1000 iterations must be enough to catch this deadlock.
        # If not, fix the bug or recheck your `random` Python module.
        for _ in range(1000):
            yield
            if stats.deadlock != 0:
                # deadlock detected
                return

        raise AssertionError("deadlock not detected")

    a = Mutex()
    b = Mutex()

    def t0() -> SimThread:
        """Direct order locking - AB."""
        yield from a.lock()
        yield from b.lock()
        yield from b.unlock()
        yield from a.unlock()

    def t1() -> SimThread:
        """Reverse order locking - BA."""
        yield from b.lock()
        yield from a.lock()
        yield from a.unlock()
        yield from b.unlock()

    stats = simsched([t0, t1], looper)
    assert stats.deadlock == 1, "must stop on the first deadlock"
    assert stats.timeout == 0, "must not timeout"
    assert stats.panic == 0, "must not panic"
