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
from typing import Callable, Mapping, Protocol, TypeAlias

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

    def _lookup(self, addr: Addr) -> int:
        """Get the value by addr."""
        # first lookup the newest store in the local storebuffer
        for a, val in reversed(self.tx.buf):
            if addr == a:
                return val

        # if not in the storebuffer, then lookup the memory
        return self.mem[addr]

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
                self.regs[reg] = self._lookup(addr)

            case (Reg() as dstreg, Reg() as srcreg):
                # local operations could be done atomic
                self.regs[dstreg] = self.regs[srcreg]

            case (Reg() as reg, int(val)):
                # local operations could be done atomic
                self.regs[reg] = val

    def mfence(self) -> SimThread:
        """Intel-like `mfence` assembly instruction."""
        yield from cond_schedule(lambda: not self.tx.buf)

    def xchg(self, dst: Addr | Reg, src: Addr | Reg) -> SimThread:
        """Intel-like x86 `xchg` assembly instruction."""
        match (dst, src):
            case (Addr(), Addr()):
                raise ValueError("x86 does not allow `xchg` from mem to mem")

            case (Addr() as a, Reg() as r) | (Reg() as r, Addr() as a):
                # xchg is implicitly locked
                yield from self.lock.lock(owner=self)

                # get both values
                aval = self._lookup(a)
                rval = self.regs[r]

                # set value to register immediately
                self.regs[r] = aval

                # set value to memory through store buffer
                yield from self.tx.send((a, rval))

                # flush the buffer and unlock
                yield from self.mfence()
                yield from self.lock.unlock()

            case (Reg() as dstreg, Reg() as srcreg):
                # local operations could be done atomic
                x, y = self.regs[dstreg], self.regs[srcreg]
                self.regs[dstreg], self.regs[srcreg] = y, x


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
            proc = HwThread(self.mem, self.lock, tx)
            self.procs.append(proc)
            self.sbctrs.append(partial(self.storebuffer, rx, proc))

    def storebuffer(self, rx: RxChannel, proc: HwThread) -> SimThread:
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
                    yield from cond_schedule(
                        lambda: not self.lock.locked or self.lock.owner is proc
                    )
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


# immutable tuple of variable values to keep it as a dict key
Snapshot: TypeAlias = tuple[tuple[str, int], ...]
SnapshotCollector = Callable[[], Snapshot]


@dataclass
class ObservedStats:
    """Class to store observed outputs."""

    storage: MutableMapping[Snapshot, int] = field(
        default_factory=lambda: collections.defaultdict(int)
    )

    def add(self, snapshot: Snapshot) -> None:
        """Add observed output to the collection."""
        self.storage[snapshot] += 1

    def __str__(self) -> str:
        """Provide human-readable string."""
        lines = []
        for snap, count in sorted(self.storage.items()):
            snapline = ", ".join(f"{reg} = {val}" for reg, val in snap)
            lines.append(f"{snapline}: {count}")

        return "\n".join(lines)


def reg_count_looper(
    outputs: ObservedStats,
    collector: SnapshotCollector,
    tso: TSO,
    stats: RunStats,
) -> LoopController:
    """Common looper for most of the TSO demos.

    Collects registers on each loop and put snapshots into storage.
    """
    yield  # first run
    while True:
        assert stats.last == SimOk(), stats.last

        snap = collector()
        outputs.add(snap)
        tso.reinit()  # explicit reinit
        yield  # next run


Config: TypeAlias = tuple[TSO, list[SimThreadConstructor], SnapshotCollector]


class Demo(Protocol):
    """Template for x86-TSO examples."""

    @staticmethod
    def configure() -> Config:
        """Run the demo and collect observed states."""
        ...

    @staticmethod
    def target() -> tuple[Snapshot, bool]:
        """The snapshot to check after the execution."""
        ...


class SbDemo:
    """SB example.

    This test shows that store buffering is observable.

    -------------------------------
    P0             | P1
    ---------------+---------------
    MOV [x] <- 1   | MOV [y] <- 1
    MOV EAX <- [y] | MOV EBX <- [x]
    -------------------------------
    Allowed Final State: P0:EAX=0 and P1:EBX=0.

    x86-TSO explanation:
    - stores to x and y are buffered in both proc0 and proc1
    - both proc0 and proc1 read x and y from memory and get zeroes
    - both storebuffers flush
    """

    @staticmethod
    def configure() -> Config:
        tso = TSO(nr_threads=2)
        p0, p1 = tso.procs

        def t0() -> SimThread:
            yield from p0.mov(Addr("x"), 1)
            yield from p0.mov(Reg("eax"), Addr("y"))

        def t1() -> SimThread:
            yield from p1.mov(Addr("y"), 1)
            yield from p1.mov(Reg("ebx"), Addr("x"))

        def snapshot() -> Snapshot:
            return (
                ("p0.eax", p0.regs[Reg("eax")]),
                ("p1.ebx", p1.regs[Reg("ebx")]),
            )

        return tso, [t0, t1], snapshot

    @staticmethod
    def target() -> tuple[Snapshot, bool]:
        """The snapshot to check after the execution."""
        return (("p0.eax", 0), ("p1.ebx", 0)), True


class IriwDemo:
    """IRIW example.

    -------------------------------------------------------------
    P0           | P1           | P2             | P3
    -------------+--------------+----------------+---------------
    MOV [x] <- 1 | MOV [y] <- 1 | MOV EAX <- [x] | MOV ECX <- [y]
                 |              | MOV EBX <- [y] | MOV EDX <- [x]
    -------------------------------------------------------------
    Forbidden Final State: P2:EAX=1 and P2:EBX=0 and P3:ECX=1 and P3:EDX=0.
    """

    @staticmethod
    def configure() -> Config:
        tso = TSO(nr_threads=4)
        p0, p1, p2, p3 = tso.procs

        def t0() -> SimThread:
            yield from p0.mov(Addr("x"), 1)

        def t1() -> SimThread:
            yield from p1.mov(Addr("y"), 1)

        def t2() -> SimThread:
            yield from p2.mov(Reg("eax"), Addr("x"))
            yield from p2.mov(Reg("ebx"), Addr("y"))

        def t3() -> SimThread:
            yield from p3.mov(Reg("ecx"), Addr("y"))
            yield from p3.mov(Reg("edx"), Addr("x"))

        def snapshot() -> Snapshot:
            return (
                ("p2.eax", p2.regs[Reg("eax")]),
                ("p2.ebx", p2.regs[Reg("ebx")]),
                ("p3.ecx", p3.regs[Reg("ecx")]),
                ("p3.edx", p3.regs[Reg("edx")]),
            )

        return tso, [t0, t1, t2, t3], snapshot

    @staticmethod
    def target() -> tuple[Snapshot, bool]:
        snap = ("p2.eax", 1), ("p2.ebx", 0), ("p3.ecx", 1), ("p3.edx", 0)
        return snap, False


class N6Demo:
    """n6 example.

    -----------------------------
    P0             | P1
    ---------------+-------------
    MOV [x] <- 1   | MOV [y] <- 2
    MOV EAX <- [x] | MOV [x] <- 2
    MOV EBX <- [y] |
    -----------------------------
    Allowed Final State: P0:EAX=1 and P0:EBX=0 and [x]=1

    x86-TSO explanation:
    - all 3 stores are buffered
    - P0 reads [x] from its own store buffer and gets 1 in EAX
    - P0 reads [y] from the memory and gets 0 in EBX
    - P1 flushes his store buffer
    - P0 finally flushes his store buffer, [x] = 1
    """

    @staticmethod
    def configure() -> Config:
        tso = TSO(nr_threads=2)
        p0, p1 = tso.procs

        def t0() -> SimThread:
            yield from p0.mov(Addr("x"), 1)
            yield from p0.mov(Reg("eax"), Addr("x"))
            yield from p0.mov(Reg("ebx"), Addr("y"))

        def t1() -> SimThread:
            yield from p1.mov(Addr("y"), 2)
            yield from p1.mov(Addr("x"), 2)

        def snapshot() -> Snapshot:
            return (
                ("p0.eax", p0.regs[Reg("eax")]),
                ("p0.ebx", p0.regs[Reg("ebx")]),
                ("[x]", tso.mem[Addr("x")]),
            )

        return tso, [t0, t1], snapshot

    @staticmethod
    def target() -> tuple[Snapshot, bool]:
        return (("p0.eax", 1), ("p0.ebx", 0), ("[x]", 1)), True


class N5Demo:
    """n5 example.

    -------------------------------
    P0             | P1
    ---------------+---------------
    MOV [x] <- 1   | MOV [x] <- 2
    MOV EAX <- [x] | MOV EBX <- [x]
    -------------------------------
    Forbidden Final State: P0:EAX=2 and P1:EBX=1
    """

    @staticmethod
    def configure() -> Config:
        tso = TSO(nr_threads=2)
        p0, p1 = tso.procs

        def t0() -> SimThread:
            yield from p0.mov(Addr("x"), 1)
            yield from p0.mov(Reg("eax"), Addr("x"))

        def t1() -> SimThread:
            yield from p1.mov(Addr("x"), 2)
            yield from p1.mov(Reg("ebx"), Addr("x"))

        def snapshot() -> Snapshot:
            return (
                ("p0.eax", p0.regs[Reg("eax")]),
                ("p1.ebx", p1.regs[Reg("ebx")]),
            )

        return tso, [t0, t1], snapshot

    @staticmethod
    def target() -> tuple[Snapshot, bool]:
        return (("p0.eax", 2), ("p1.ebx", 1)), False


class N4bDemo:
    """n4b example.

    -------------------------------
    P0             | P1
    ---------------+---------------
    MOV EAX <- [x] | MOV ECX <- [x]
    MOV [x] <- 1   | MOV [x] <- 2
    -------------------------------
    Forbidden Final State: P0:EAX=2 and P1:ECX=1
    """

    @staticmethod
    def configure() -> Config:
        tso = TSO(nr_threads=2)
        p0, p1 = tso.procs

        def t0() -> SimThread:
            yield from p0.mov(Reg("eax"), Addr("x"))
            yield from p0.mov(Addr("x"), 1)

        def t1() -> SimThread:
            yield from p1.mov(Reg("ecx"), Addr("x"))
            yield from p1.mov(Addr("x"), 2)

        def snapshot() -> Snapshot:
            return (
                ("p0.eax", p0.regs[Reg("eax")]),
                ("p1.ecx", p1.regs[Reg("ecx")]),
            )

        return tso, [t0, t1], snapshot

    @staticmethod
    def target() -> tuple[Snapshot, bool]:
        return (("p0.eax", 2), ("p1.ecx", 1)), False


class Ex_8_1_Demo:
    """Example 8-1.

    Stores are not reoredered with other stores.

    -----------------------------
    P0           | P1
    -------------+---------------
    MOV [x] <- 1 | MOV EAX <- [y]
    MOV [y] <- 1 | MOV EBX <- [x]
    -------------------------------
    Forbidden Final State: P1:EAX=1 and P1:EBX=0.
    """

    @staticmethod
    def configure() -> Config:
        tso = TSO(nr_threads=2)
        p0, p1 = tso.procs

        def t0() -> SimThread:
            yield from p0.mov(Addr("x"), 1)
            yield from p0.mov(Addr("y"), 1)

        def t1() -> SimThread:
            yield from p1.mov(Reg("eax"), Addr("y"))
            yield from p1.mov(Reg("ebx"), Addr("x"))

        def snapshot() -> Snapshot:
            return (
                ("p1.eax", p1.regs[Reg("eax")]),
                ("p1.ebx", p1.regs[Reg("ebx")]),
            )

        return tso, [t0, t1], snapshot

    @staticmethod
    def target() -> tuple[Snapshot, bool]:
        return (("p1.eax", 1), ("p1.ebx", 0)), False


class Ex_8_2_Demo:
    """Example 8-2.

    Stores are not reoredered with older loads.

    -------------------------------
    P0             | P1
    ---------------+---------------
    MOV EAX <- [x] | MOV EBX <- [y]
    MOV [y] <- 1   | MOV [x] <- 1
    -------------------------------
    Forbidden Final State: P0:EAX=1 and P1:EBX=1
    """

    @staticmethod
    def configure() -> Config:
        tso = TSO(nr_threads=2)
        p0, p1 = tso.procs

        def t0() -> SimThread:
            yield from p0.mov(Reg("eax"), Addr("x"))
            yield from p0.mov(Addr("y"), 1)

        def t1() -> SimThread:
            yield from p1.mov(Reg("ebx"), Addr("y"))
            yield from p1.mov(Addr("x"), 1)

        def snapshot() -> Snapshot:
            return (
                ("p0.eax", p0.regs[Reg("eax")]),
                ("p1.ebx", p1.regs[Reg("ebx")]),
            )

        return tso, [t0, t1], snapshot

    @staticmethod
    def target() -> tuple[Snapshot, bool]:
        return (("p0.eax", 1), ("p1.ebx", 1)), False


class Ex_8_4_Demo:
    """Example 8-4.

    Loads are not reordered with older stores to the same location.

    ---------------
    P0
    ---------------
    MOV [x] <- 1
    MOV EAX <- [x]
    ---------------
    Required Final State: P0:EAX=1
    """

    @staticmethod
    def configure() -> Config:
        tso = TSO(nr_threads=1)
        [p0] = tso.procs

        def t0() -> SimThread:
            yield from p0.mov(Addr("x"), 1)
            yield from p0.mov(Reg("eax"), Addr("x"))

        def snapshot() -> Snapshot:
            return (("p0.eax", p0.regs[Reg("eax")]),)

        return tso, [t0], snapshot

    @staticmethod
    def target() -> tuple[Snapshot, bool]:
        return (("p0.eax", 0),), False


class Ex_8_6_Demo:
    """Example 8-6.

    Stores are transitively visible.

    ----------------------------------------------
    P0           | P1             | P2
    -------------+----------------+---------------
    MOV [x] <- 1 | MOV EAX <- [x] | MOV EBX <- [y]
                 | MOV [y] <- 1   | MOV ECX <- [x]
    ----------------------------------------------
    Forbidden Final State: P1:EAX=1 and P2:EBX=1 and P2:ECX=0
    """

    @staticmethod
    def configure() -> Config:
        tso = TSO(nr_threads=3)
        p0, p1, p2 = tso.procs

        def t0() -> SimThread:
            yield from p0.mov(Addr("x"), 1)

        def t1() -> SimThread:
            yield from p1.mov(Reg("eax"), Addr("x"))
            yield from p1.mov(Addr("y"), 1)

        def t2() -> SimThread:
            yield from p2.mov(Reg("ebx"), Addr("y"))
            yield from p2.mov(Reg("ecx"), Addr("x"))

        def snapshot() -> Snapshot:
            return (
                ("p1.eax", p1.regs[Reg("eax")]),
                ("p2.ebx", p2.regs[Reg("ebx")]),
                ("p2.ecx", p2.regs[Reg("ecx")]),
            )

        return tso, [t0, t1, t2], snapshot

    @staticmethod
    def target() -> tuple[Snapshot, bool]:
        return (("p1.eax", 1), ("p2.ebx", 1), ("p3.ecx", 1)), False


class Ex_8_9_Demo:
    """Example 8-9.

    Loads are not reordered with locks.

    ---------------------------------
    P0              | P1
    ----------------+----------------
    XCHG [x] <- EAX | XCHG [y] <- ECX
    MOV  EBX <- [y] | MOV  EDX <- [x]
    ---------------------------------
    Initial state: P0:EAX=1 and P1:ECX=1
    Forbidden Final State: P0:EBX=0 and P1:EDX=0
    """

    @staticmethod
    def configure() -> Config:
        tso = TSO(nr_threads=2)
        p0, p1 = tso.procs

        def t0() -> SimThread:
            p0.regs[Reg("eax")] = 1  # initialize
            yield from p0.xchg(Addr("x"), Reg("eax"))
            yield from p0.mov(Reg("ebx"), Addr("y"))

        def t1() -> SimThread:
            p1.regs[Reg("ecx")] = 1  # initialize
            yield from p1.xchg(Addr("y"), Reg("ecx"))
            yield from p1.mov(Reg("edx"), Addr("x"))

        def snapshot() -> Snapshot:
            return (
                ("p0.ebx", p0.regs[Reg("ebx")]),
                ("p1.edx", p1.regs[Reg("edx")]),
            )

        return tso, [t0, t1], snapshot

    @staticmethod
    def target() -> tuple[Snapshot, bool]:
        return (("p0.ebx", 0), ("p1.edx", 0)), False


class Amd5Demo:
    """AMD5 example.

    -------------------------------
    P0             | P1
    ---------------+---------------
    MOV [x] <- 1   | MOV [y] <- 1
    MFENCE         | MFENCE
    MOV EAX <- [y] | MOV EBX <- [x]
    -------------------------------
    Forbidden Final State: P0:EAX=0 and P1:EBX=0.
    """

    @staticmethod
    def configure() -> Config:
        tso = TSO(nr_threads=2)
        p0, p1 = tso.procs

        def t0() -> SimThread:
            yield from p0.mov(Addr("x"), 1)
            yield from p0.mfence()
            yield from p0.mov(Reg("eax"), Addr("y"))

        def t1() -> SimThread:
            yield from p1.mov(Addr("y"), 1)
            yield from p1.mfence()
            yield from p1.mov(Reg("ebx"), Addr("x"))

        def snapshot() -> Snapshot:
            return (
                ("p0.eax", p0.regs[Reg("eax")]),
                ("p1.ebx", p1.regs[Reg("ebx")]),
            )

        return tso, [t0, t1], snapshot

    @staticmethod
    def target() -> tuple[Snapshot, bool]:
        """The snapshot to check after the execution."""
        return (("p0.eax", 0), ("p1.ebx", 0)), False


def play_demo(demo: type[Demo]) -> bool:
    """Play the demo from the template."""
    # prepare the demo
    tso, thrctrs, snapshot = demo.configure()

    print(demo.__doc__)
    print("Running the simulator, hit Ctrl+C to stop...")

    # run the demo and collect the observed states
    outputs = ObservedStats()
    simsched(
        itertools.chain(
            [partial(p.wrap, t) for p, t in zip(tso.procs, thrctrs)],
            tso.sbctrs,
        ),
        loopctr=partial(reg_count_looper, outputs, snapshot, tso),
    )

    print("interrupted\n")
    print(f"Observed states (total {len(outputs.storage)}):")
    print(outputs)

    # investigate the target snapshot
    target, allowed = demo.target()
    status = "Allowed" if allowed else "Forbidden"
    count = outputs.storage.get(target, 0)
    print(f"{status} state: {target} happened {count} times")

    return allowed == bool(count)


DEMOS: Mapping[str, type[Demo]] = {
    "sb": SbDemo,
    "iriw": IriwDemo,
    "n6": N6Demo,
    "n5": N5Demo,
    "n4b": N4bDemo,
    "ex8-1": Ex_8_1_Demo,
    "ex8-2": Ex_8_2_Demo,
    "ex8-4": Ex_8_4_Demo,
    "ex8-6": Ex_8_6_Demo,
    "ex8-9": Ex_8_9_Demo,
    "amd5": Amd5Demo,
}


def main() -> None:
    """CLI entrypoint."""
    prog, *args = sys.argv
    try:
        match args:
            case [name]:
                if name not in DEMOS:
                    raise NotImplementedError(name)

                success = play_demo(DEMOS[name])
            case _:
                raise ValueError
    except NotImplementedError as nie:
        print(f"{nie.args[0]} not supported, select from ({'|'.join(DEMOS)})")
        sys.exit(1)
    except ValueError:
        print(f"usage: {prog} ({'|'.join(DEMOS)})")
        sys.exit(1)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
