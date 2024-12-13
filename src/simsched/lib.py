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

"""Collection of scheduling/synchronization primitives."""

import collections
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from simsched.core import SimThread, cond_schedule, schedule

Item = TypeVar("Item")


@dataclass
class Cell(Generic[Item]):
    """Mutable container for any type."""

    val: Item


@dataclass
class Mutex:
    """Simple mutual exclusion primitive."""

    locked: bool = False
    label: str | None = None
    owner: Any = None

    def lock(self, *, owner: Any = None) -> SimThread:
        """Try to aquire the lock."""
        yield from cond_schedule(lambda: not self.locked)
        self.locked = True
        self.owner = owner

    def unlock(self) -> SimThread:
        """Release the lock and yield the execution.

        If not locked, the exception is raised.
        """
        if not self.locked:
            raise RuntimeError("trying to unlock non-locked mutex")

        self.locked = False
        self.owner = None
        yield from schedule()


@dataclass
class TxChannel:
    """Generic one-way TX channel for passing objects of any type."""

    buf: collections.deque[Any]

    def send(self, item: Any) -> SimThread:
        """Put the item to the channel."""
        self.buf.append(item)
        yield from schedule()


@dataclass
class RxChannel:
    """Generic one-way RX channel for receiving objects of any type."""

    buf: collections.deque[Any]

    def peek(self, ret: Cell[Any]) -> SimThread:
        """Block until item appears in the channel."""
        yield from cond_schedule(lambda: bool(self.buf))
        ret.val = self.buf[0]

    def consume(self) -> None:
        """Consume the current item."""
        self.buf.popleft()

    def recv(self, ret: Cell[Any]) -> SimThread:
        """Get the item from the channel if present and block otherwise."""
        yield from self.peek(ret)
        self.consume()


def create_channel() -> tuple[TxChannel, RxChannel]:
    """Create pair of channels."""
    buf = collections.deque()  # create shared buffer
    return TxChannel(buf), RxChannel(buf)
