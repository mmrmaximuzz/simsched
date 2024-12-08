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

from dataclasses import dataclass

from simsched.core import SimThread, cond_schedule, schedule


@dataclass
class Mutex:
    """Simple mutual exclusion primitive."""

    locked: bool = False

    def lock(self) -> SimThread:
        """Try to aquire the lock."""
        yield from cond_schedule(lambda: not self.locked)
        self.locked = True

    def unlock(self) -> SimThread:
        """Release the lock and yield the execution."""
        self.locked = False

        # Immediately return the control to the scheduler to give other threads
        # a chance to compete for this lock.
        yield from schedule()
