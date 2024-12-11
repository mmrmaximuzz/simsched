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
from simsched.engine import SimDeadlock, SimOk, SimPanic, run
from simsched.lib import Cell, Mutex, create_channel


def test_mutex_single_lock():
    """Single lock must work."""
    mtx = Mutex()

    def thread() -> SimThread:
        """Simulate just one lock operation."""
        yield from mtx.lock()

    assert run([thread]) == SimOk(), "must not deadlock"
    assert mtx.locked, "must be locked"


def test_mutex_single_unlock():
    """Single unlock must work."""
    mtx = Mutex(locked=True)

    def thread() -> SimThread:
        """Simulate just one unlock operation."""
        yield from mtx.unlock()

    assert run([thread]) == SimOk(), "must not deadlock"
    assert not mtx.locked, "must be unlocked"


def test_mutex_lock_unlock():
    """Single thread must be fine with the lock."""
    mtx = Mutex()

    def thread() -> SimThread:
        """Simulate just one lock/unlock sequence."""
        yield from mtx.lock()
        yield from mtx.unlock()

    assert run([thread]) == SimOk(), "must not deadlock"
    assert not mtx.locked, "must be unlocked"


def test_mutex_self_deadlock():
    """Single thread must deadlock if Mutex is acquired twice."""
    mtx = Mutex()

    def thread() -> SimThread:
        """Simulate self-deadlock."""
        yield from mtx.lock()
        yield from mtx.lock()  # deadlock

    assert run([thread]) == SimDeadlock(), "must deadlock"
    assert mtx.locked, "must be locked forever"


def test_mutex_counter_data_race():
    """Mutex must protect from simple data races."""
    mtx = Mutex()
    cell: Cell[int] = Cell(0)

    def thread() -> SimThread:
        """Simulate non-atomic counter increment."""
        yield from mtx.lock()
        val = cell.val
        yield from schedule()  # simulate non-atomic RMW operation
        cell.val = val + 1
        yield from mtx.unlock()

    nr_threads = 10
    assert run(thread for _ in range(nr_threads)) == SimOk(), "must finish"
    assert cell.val == nr_threads, "have correct value"


def test_mutex_unlock_when_not_locked():
    """Must raise panic when unlocking free mutex."""
    mtx = Mutex()

    def thread() -> SimThread:
        """Simulate self-deadlock."""
        yield from mtx.unlock()

    res = run([thread])
    assert isinstance(res, SimPanic), "must got PANIC"
    assert isinstance(res.e, RuntimeError), "must get exact type"
    assert not mtx.locked, "must remain unlocked"


def test_channel_send():
    """Channels must keep sent data."""
    tx, _ = create_channel()

    def thread() -> SimThread:
        """Simple sender thread."""
        yield from tx.send("hello")

    assert run([thread]) == SimOk(), "must complete"
    assert list(tx.buf) == ["hello"], "must contain a value"


def test_channel_multisend():
    """Channels must keep all sent data."""
    tx, _ = create_channel()

    def thread() -> SimThread:
        """Simple sender thread."""
        yield from tx.send(100)
        yield from tx.send(200)

    assert run([thread]) == SimOk(), "must complete"
    assert list(tx.buf) == [100, 200], "must contain all values"


def test_channel_both():
    """Channels must transmit data."""
    tx, rx = create_channel()

    def sender() -> SimThread:
        """Simple sender thread."""
        yield from tx.send("hello")

    def receiver() -> SimThread:
        """Simple receiver thread."""
        cell: Cell[str] = Cell("")
        yield from rx.recv(cell)
        assert cell.val == "hello", "must get correct message"

    assert run([sender, receiver]) == SimOk(), "must complete"


def test_channel_recv_blocks():
    """RxChannel must block when no data available."""
    steps = [False, False]
    _, rx = create_channel()

    def thread() -> SimThread:
        """Emulate blocking."""
        steps[0] = True
        yield from rx.recv(Cell(None))
        steps[1] = True

    assert run([thread]) == SimDeadlock(), "must block forever"
    assert steps == [True, False], "must not continue after blocking recv"
