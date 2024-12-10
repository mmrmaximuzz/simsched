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

from simsched.core import SimThread, schedule
from simsched.engine import SimDeadlock, SimOk
from simsched.lib import Mutex
from simsched.runner import (
    LoopController,
    RunStats,
    simsched,
    time_report_looper,
)


def test_simsched_simple():
    """Runner must be controlled by the looper object."""
    nr_iters = 5

    def looper(stats: RunStats) -> LoopController:
        """Simple looper with upper limit of cycles."""
        while stats.total < nr_iters:
            yield

    def thread() -> SimThread:
        """Simple thread to execute."""
        yield from schedule()

    stats = simsched([thread], looper)
    assert stats == RunStats(
        total=nr_iters,
        ok=nr_iters,
        deadlock=0,
        timeout=0,
        panic=0,
        last=SimOk(),
    )


def test_simsched_abba_deadlock():
    """Runner must catch the simple deadlock."""

    def looper(stats: RunStats) -> LoopController:
        """Loop until deadlock is discovered with some upper limit."""
        # 1000 iterations must be enough to catch this deadlock.
        # If not, fix the bug or recheck your `random` Python module.
        while stats.total < 1000:
            yield
            if stats.last == SimDeadlock():
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
    assert stats.total > 0, "at least one run"
    assert stats.deadlock == 1, "must stop on the first deadlock"
    assert stats.timeout == 0, "must not timeout"
    assert stats.panic == 0, "must not panic"


def test_time_report_looper():
    """Time based looper must call report."""
    # simulate capturing environment for reporter function
    env = {
        "times": [],
        "oks": [],
    }

    def reporter(time: float, stats: RunStats) -> None:
        """Reporter collects OK results for 3 periods."""
        env["times"].append(time)
        env["oks"].append(stats.ok)
        if len(env["times"]) >= 3:
            raise KeyboardInterrupt  # simched must stop on KeyboardInterrupt

    looper = partial(time_report_looper, reporter, 0.01)

    def thread() -> SimThread:
        """Do nothing."""
        yield from schedule()

    simsched([thread], looper)
    assert len(env["oks"]) == 3
    assert env["oks"][0] < env["oks"][1] < env["oks"][2], "oks must increase"
    assert all(t >= 0.01 for t in env["times"]), "must keep interval"
