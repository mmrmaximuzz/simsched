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

"""This is a demo example for the simsched project.

Classic example for incrementing non-atomic counter from different threads
could lead to pretty unpredictable results based on the scheduler behavior.

This example collects the observed counter values in a simulation.
"""

import collections
from dataclasses import dataclass
from functools import partial
import sys

from simsched.core import SimThread, schedule
from simsched.runner import LoopController, RunStats, simsched


@dataclass
class Cell:
    """Mutable int wrapper."""

    val: int = 0


def bad_inc_thread(counter: Cell, nr_incs: int) -> SimThread:
    """Non-atomic incrementer."""
    for _ in range(nr_incs):
        val = counter.val
        yield from schedule()
        counter.val = val + 1


def demo(nr_threads: int = 5, nr_incs: int = 3) -> None:
    """Run non-atomic counter increment demo with params."""
    # create environment to keep between simulation runs
    counter = Cell()
    outputs = collections.defaultdict(int)

    # define the looper object to control the execution loop
    def looper(stats: RunStats) -> LoopController:
        """Controller for the execution."""
        yield  # first run
        while True:
            if stats.total % 5000 == 0:
                print(f"total: {stats.total}, outputs: {len(outputs)}")

            # collect the current output result
            outputs[counter.val] += 1

            # reset the counter value before next run
            counter.val = 0
            yield  # schedule next run

    print("Demo - non-atomic counter")
    print(f"threads: {nr_threads}, increments: {nr_incs}")
    print("Hit Ctrl+C to stop simulation")
    print("Running...")

    # run the simsched tool
    thrdctr = partial(bad_inc_thread, counter, nr_incs)
    simsched((thrdctr for _ in range(nr_threads)), looper)

    print()
    print("The following counter values have been observed")
    for val, count in sorted(outputs.items()):
        print(f"{val:2}: {count}")


def main() -> None:
    """CLI entrypoint."""
    prog, *args = sys.argv
    try:
        match args:
            case []:
                demo()
            case [nr_threads]:
                demo(int(nr_threads))
            case [nr_threads, nr_incs]:
                demo(int(nr_threads), int(nr_incs))
            case _:
                raise ValueError
    except Exception:
        print(f"usage: {prog} [THREADS [INCREMENTS]]")
        sys.exit(1)


if __name__ == "__main__":
    main()
