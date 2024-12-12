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

Almost all current and widely used processor architectures (such as x86*,
powerpc, arm) exhibit some kind of relaxed (or weak) memory models. Due to
that, some observable behaviors of some parallel programs for these
architectures cannot be represented as a result of any interleaving of the
original program instructions. This demo implements x86-TSO, which is a formal
model of execution for x86* architecture, and it tries to reproduce some of the
common multithreading issues that could happen on this platform.

The issues for this demo are deliberately taken from this must-read paper.
https://www.cl.cam.ac.uk/~pes20/weakmemory/cacm.pdf
"""

import collections
import sys
import itertools
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from functools import partial
from typing import TypeAlias

from simsched.core import SimThread, cond_schedule
from simsched.engine import SimOk, SimThreadConstructor
from simsched.lib import Cell, Mutex, RxChannel, TxChannel, create_channel
from simsched.runner import LoopController, RunStats, simsched


# Use real dataclasses instead of typing.NewType for pattern matching
@dataclass(frozen=True)
class Addr:
    addr: str


@dataclass(frozen=True)
class Reg:
    reg: str


Memory: TypeAlias = MutableMapping[Addr, int]
Registers: TypeAlias = MutableMapping[Reg, int]


@dataclass
class HwThread:
    """Single execution unit in the TSO model."""

    mem: Memory
    lock: Mutex
    tx: TxChannel
    regs: Registers = field(default_factory=dict)

    def wrap(self, t: SimThreadConstructor) -> SimThread:
        """Wrap the thread with closing storebuffer thread."""
        yield from t()
        yield from self.tx.send(None)

    def mov(self, dst: Addr | Reg, src: Addr | Reg | int) -> SimThread:
        """Intel-like x86 `mov` assembly instruction."""
        match (dst, src):
            case (Addr(), Addr()):
                raise ValueError("x86 does not allow `mov` from mem to mem")

            case (Addr() as addr, Reg() as reg):
                # store operations go through store buffer
                val = self.regs[reg]
                yield from self.tx.send((addr, val))

            case (Addr() as addr, int(val)):
                # store operations go through store buffer
                yield from self.tx.send((addr, val))

            case (Reg() as reg, Addr() as addr):
                # load operations are allowed only when memory is not locked
                yield from cond_schedule(lambda: not self.lock.locked)

                # first lookup the newest store in the local storebuffer
                for a, val in reversed(self.tx.buf):
                    if addr == a:
                        self.regs[reg] = val
                        return

                # if not in the storebuffer, then lookup the memory
                self.regs[reg] = self.mem[addr]

            case (Reg() as dstreg, Reg() as srcreg):
                # local operations could be done atomic
                self.regs[dstreg] = self.regs[srcreg]

            case (Reg() as reg, int(val)):
                # local operations could be done atomic
                self.regs[reg] = val


@dataclass
class TSO:
    """Class representing TSO memory model state."""

    mem: Memory
    lock: Mutex
    sbctrs: list[SimThreadConstructor]
    procs: list[HwThread]

    def __init__(self, *, nr_threads: int) -> None:
        """Class constructor."""
        self.mem = collections.defaultdict(int)
        self.lock = Mutex()
        self.sbctrs = []
        self.procs = []
        for _ in range(nr_threads):
            tx, rx = create_channel()
            self.sbctrs.append(partial(self.storebuffer, rx))
            self.procs.append(HwThread(self.mem, self.lock, tx))

    def storebuffer(self, rx: RxChannel) -> SimThread:
        """Pseudo-thread for flushing store buffer."""
        # memory store instruction or finish signal
        cell: Cell[tuple[Addr, int] | None] = Cell(None)

        while True:
            # Do not consume the value, just peek. We need to store the value
            # in the buffer as hwthread may try to read from pending stores.
            yield from rx.peek(cell)
            match cell.val:
                case (addr, value):
                    # can flush values only when the memory is not locked
                    yield from cond_schedule(lambda: not self.lock.locked)
                    self.mem[addr] = value
                    rx.consume()  # now the value is flushed, we can remove it
                case None:
                    # finish signal, done
                    rx.consume()
                    return


    def reinit(self) -> None:
        """Reinit TSO state by freeing all written memory."""
        self.mem.clear()
        for p in self.procs:
            assert not p.tx.buf, "must be empty when finished"
            p.regs.clear()


# immutable tuple of register values to keep it as a dict key
RegState: TypeAlias = tuple[tuple[str, int], ...]


def sb() -> None:
    """Store buffer example.

    This test shows that store buffering is observable.

    -------------------------------
    Proc 0         | Proc 1
    ---------------+---------------
    MOV [x] <- 1   | MOV [y] <- 1
    MOV EAX <- [y] | MOV EBX <- [x]
    -------------------------------
    Allowed Final State: Proc 0: EAX = 0 and Proc 1: EBX = 0.

    The TSO explanation:
    - stores to x and y are buffered in both proc0 and proc1
    - both proc0 and proc1 read x and y from memory and get zeroes
    - both storebuffers flush
    """
    tso = TSO(nr_threads=2)
    p0, p1 = tso.procs

    def t0() -> SimThread:
        """Proc 0 for `sb` example."""
        yield from p0.mov(Addr("x"), 1)
        yield from p0.mov(Reg("eax"), Addr("y"))

    def t1() -> SimThread:
        """Proc 1 for `sb` example."""
        yield from p1.mov(Addr("y"), 1)
        yield from p1.mov(Reg("ebx"), Addr("x"))

    outputs: MutableMapping[RegState, int] = collections.defaultdict(int)

    def looper(stats: RunStats) -> LoopController:
        """Collect the possible outputs."""
        yield  # first run
        while True:
            assert stats.last == SimOk(), stats.last

            outputs[
                (
                    ("p0.eax", p0.regs[Reg("eax")]),
                    ("p1.ebx", p1.regs[Reg("ebx")]),
                )
            ] += 1

            # reinit TSO before next run
            tso.reinit()
            yield

    print("Demo - sb")
    print(sb.__doc__)
    print("Running the simulator, hit Ctrl+C to stop...")

    simsched(
        itertools.chain(
            [partial(p0.wrap, t0), partial(p1.wrap, t1)],
            tso.sbctrs,
        ),
        looper,
    )

    print("interrupted")
    for state, count in sorted(outputs.items()):
        for reg, val in state:
            print(reg, val, end=" ")
        print(":", count)

    allowed = (("p0.eax", 0), ("p1.ebx", 0))
    if count := outputs.get(allowed, 0):
        print(f"allowed state: {allowed} happened {count} times")


DEMOS = {
    "sb": sb,
}


def main() -> None:
    """CLI entrypoint."""
    prog, *args = sys.argv
    try:
        match args:
            case [name]:
                if name not in DEMOS:
                    raise NotImplementedError(name)

                DEMOS[name]()
            case _:
                raise ValueError
    except NotImplementedError as nie:
        print(f"{nie.args[0]} not supported, select from ({'|'.join(DEMOS)})")
        sys.exit(1)
    except ValueError:
        print(f"usage: {prog} ({'|'.join(DEMOS)})")
        sys.exit(1)


if __name__ == "__main__":
    main()
