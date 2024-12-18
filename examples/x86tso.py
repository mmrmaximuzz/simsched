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

The issues for this demo are deliberately taken from these must-read papers.
https://www.cl.cam.ac.uk/~pes20/weakmemory/cacm.pdf
https://www.cl.cam.ac.uk/~pes20/weakmemory/ecoop10.pdf
"""

import collections
import contextlib
import itertools
import sys
from collections.abc import Iterable, MutableMapping
from dataclasses import dataclass, field
from functools import partial
from io import StringIO
from typing import Callable, Mapping, Protocol, Self, TypeAlias

from simsched.core import SimThread, cond_schedule, schedule
from simsched.engine import SimThreadConstructor
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
class Processor:
    """Single execution unit in the TSO model."""

    mem: Memory
    lock: Mutex
    tx: TxChannel
    regs: Registers = field(default_factory=dict)

    def wrap(self, t: Callable[[Self], SimThread]) -> SimThread:
        """Wrap the program with closing storebuffer thread."""
        yield from t(self)
        yield from self.tx.send(None)

    def _lookup(self, addr: Addr) -> int:
        """Get the value by addr."""
        # first lookup the newest store in the local storebuffer
        for a, val in reversed(self.tx.buf):
            if addr == a:
                return val

        # if not in the storebuffer, then lookup the memory
        return self.mem[addr]

    def _memlock(self) -> SimThread:
        """Lock the memory system."""
        yield from self.lock.lock(owner=self)

    def _memunlock(self) -> SimThread:
        """Unlock the memory system."""
        yield from self.mfence()  # flushing the self store buffer is a must
        yield from self.lock.unlock()

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
                yield from self._memlock()

                # get both values
                aval = self._lookup(a)
                rval = self.regs[r]

                # set value to register immediately, and send to memory
                self.regs[r] = aval
                yield from self.tx.send((a, rval))

                yield from self._memunlock()

            case (Reg() as dstreg, Reg() as srcreg):
                # local operations could be done atomic
                x, y = self.regs[dstreg], self.regs[srcreg]
                self.regs[dstreg], self.regs[srcreg] = y, x

    def inc(self, location: Addr) -> SimThread:
        """Non-atomic memory increment."""
        assert Reg("tmp") not in self.regs, "sanity check"

        yield from self.mov(Reg("tmp"), location)
        self.regs[Reg("tmp")] += 1
        yield from self.mov(location, Reg("tmp"))

        self.regs.pop(Reg("tmp"))  # cleanup

    def lock_xadd(self, location: Addr, reg: Reg) -> SimThread:
        """Atomic memory addition."""
        yield from self._memlock()

        old = self._lookup(location)
        yield from self.mov(location, old + self.regs[reg])

        yield from self._memunlock()

        # return the old value in the register
        self.regs[reg] = old

    def lock_dec(self, location: Addr, ret: Cell[int]) -> SimThread:
        """Atomic memory decrement."""
        yield from self._memlock()

        old = self._lookup(location)
        yield from self.mov(location, old - 1)

        yield from self._memunlock()

        # return the old value in return value (for CMP after return)
        ret.val = old


@dataclass
class TSO:
    """Class representing TSO memory model state."""

    mem: Memory
    lock: Mutex
    sbctrs: list[SimThreadConstructor]
    procs: list[Processor]
    defmem: Mapping[Addr, int]

    def __init__(
        self,
        *,
        nr_threads: int,
        defmem: Mapping[Addr, int] = {},
    ) -> None:
        """Class constructor."""
        self.mem = collections.defaultdict(int)
        self.lock = Mutex()
        self.sbctrs = []
        self.procs = []
        self.defmem = defmem

        # create processors and storebuffers
        for _ in range(nr_threads):
            tx, rx = create_channel()
            proc = Processor(self.mem, self.lock, tx)
            self.procs.append(proc)
            self.sbctrs.append(partial(self.storebuffer, rx, proc))

        # initialize memory
        self.mem.update(self.defmem)

    def storebuffer(self, rx: RxChannel, proc: Processor) -> SimThread:
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
        """Reinit TSO state."""
        self.mem.clear()  # clear all state
        self.mem.update(self.defmem)  # reinitialize
        for p in self.procs:
            assert not p.tx.buf, "must be empty when finished"
            p.regs.clear()


# immutable tuple of variable values to keep it as a dict key
Snapshot: TypeAlias = tuple[int, ...]
NamedSnapshot: TypeAlias = tuple[tuple[str, int], ...]
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

    def describe(self, inames: Iterable[str]) -> str:
        """Provide human-readable string."""
        names = list(inames)

        lines = []
        for snap, count in sorted(self.storage.items()):
            namedsnap = zip(names, snap)
            snapline = ", ".join(f"{name} = {val}" for name, val in namedsnap)
            lines.append(f"{snapline}: {count}")

        return "\n".join(lines)


def reg_count_looper(
    outputs: ObservedStats,
    collector: SnapshotCollector,
    tso: TSO,
    iters: int,
    stats: RunStats,
) -> LoopController:
    """Common looper for most of the TSO demos.

    Collects registers on each loop and put snapshots into storage.
    """
    yield  # first run
    while True:
        snap = collector()
        outputs.add(snap)
        tso.reinit()  # explicit reinit

        # if limit is provided, then check
        if iters and stats.total >= iters:
            return

        yield  # next run


Prog: TypeAlias = Callable[[Processor], SimThread]
Config: TypeAlias = tuple[TSO, list[Prog], SnapshotCollector]
Target: TypeAlias = tuple[NamedSnapshot, bool]


class Demo(Protocol):
    """Template for x86-TSO examples."""

    @staticmethod
    def configure() -> Config:
        """Run the demo and collect observed states."""
        ...

    @staticmethod
    def target() -> Target:
        """The snapshot target to check after the execution."""
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

        def t0(p: Processor) -> SimThread:
            yield from p.mov(Addr("x"), 1)
            yield from p.mov(Reg("eax"), Addr("y"))

        def t1(p: Processor) -> SimThread:
            yield from p.mov(Addr("y"), 1)
            yield from p.mov(Reg("ebx"), Addr("x"))

        def snapshot() -> Snapshot:
            return (
                p0.regs[Reg("eax")],
                p1.regs[Reg("ebx")],
            )

        return tso, [t0, t1], snapshot

    @staticmethod
    def target() -> Target:
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
        _, _, p2, p3 = tso.procs

        def t0(p: Processor) -> SimThread:
            yield from p.mov(Addr("x"), 1)

        def t1(p: Processor) -> SimThread:
            yield from p.mov(Addr("y"), 1)

        def t2(p: Processor) -> SimThread:
            yield from p.mov(Reg("eax"), Addr("x"))
            yield from p.mov(Reg("ebx"), Addr("y"))

        def t3(p: Processor) -> SimThread:
            yield from p.mov(Reg("ecx"), Addr("y"))
            yield from p.mov(Reg("edx"), Addr("x"))

        def snapshot() -> Snapshot:
            return (
                p2.regs[Reg("eax")],
                p2.regs[Reg("ebx")],
                p3.regs[Reg("ecx")],
                p3.regs[Reg("edx")],
            )

        return tso, [t0, t1, t2, t3], snapshot

    @staticmethod
    def target() -> Target:
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
        p0, _ = tso.procs

        def t0(p: Processor) -> SimThread:
            yield from p.mov(Addr("x"), 1)
            yield from p.mov(Reg("eax"), Addr("x"))
            yield from p.mov(Reg("ebx"), Addr("y"))

        def t1(p: Processor) -> SimThread:
            yield from p.mov(Addr("y"), 2)
            yield from p.mov(Addr("x"), 2)

        def snapshot() -> Snapshot:
            return (
                p0.regs[Reg("eax")],
                p0.regs[Reg("ebx")],
                tso.mem[Addr("x")],
            )

        return tso, [t0, t1], snapshot

    @staticmethod
    def target() -> Target:
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

        def t0(p: Processor) -> SimThread:
            yield from p.mov(Addr("x"), 1)
            yield from p.mov(Reg("eax"), Addr("x"))

        def t1(p: Processor) -> SimThread:
            yield from p.mov(Addr("x"), 2)
            yield from p.mov(Reg("ebx"), Addr("x"))

        def snapshot() -> Snapshot:
            return (
                p0.regs[Reg("eax")],
                p1.regs[Reg("ebx")],
            )

        return tso, [t0, t1], snapshot

    @staticmethod
    def target() -> Target:
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

        def t0(p: Processor) -> SimThread:
            yield from p.mov(Reg("eax"), Addr("x"))
            yield from p.mov(Addr("x"), 1)

        def t1(p: Processor) -> SimThread:
            yield from p.mov(Reg("ecx"), Addr("x"))
            yield from p.mov(Addr("x"), 2)

        def snapshot() -> Snapshot:
            return (
                p0.regs[Reg("eax")],
                p1.regs[Reg("ecx")],
            )

        return tso, [t0, t1], snapshot

    @staticmethod
    def target() -> Target:
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
        _, p1 = tso.procs

        def t0(p: Processor) -> SimThread:
            yield from p.mov(Addr("x"), 1)
            yield from p.mov(Addr("y"), 1)

        def t1(p: Processor) -> SimThread:
            yield from p.mov(Reg("eax"), Addr("y"))
            yield from p.mov(Reg("ebx"), Addr("x"))

        def snapshot() -> Snapshot:
            return (
                p1.regs[Reg("eax")],
                p1.regs[Reg("ebx")],
            )

        return tso, [t0, t1], snapshot

    @staticmethod
    def target() -> Target:
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

        def t0(p: Processor) -> SimThread:
            yield from p.mov(Reg("eax"), Addr("x"))
            yield from p.mov(Addr("y"), 1)

        def t1(p: Processor) -> SimThread:
            yield from p.mov(Reg("ebx"), Addr("y"))
            yield from p.mov(Addr("x"), 1)

        def snapshot() -> Snapshot:
            return (
                p0.regs[Reg("eax")],
                p1.regs[Reg("ebx")],
            )

        return tso, [t0, t1], snapshot

    @staticmethod
    def target() -> Target:
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

        def t0(p: Processor) -> SimThread:
            yield from p.mov(Addr("x"), 1)
            yield from p.mov(Reg("eax"), Addr("x"))

        def snapshot() -> Snapshot:
            return (p0.regs[Reg("eax")],)

        return tso, [t0], snapshot

    @staticmethod
    def target() -> Target:
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
        _, p1, p2 = tso.procs

        def t0(p: Processor) -> SimThread:
            yield from p.mov(Addr("x"), 1)

        def t1(p: Processor) -> SimThread:
            yield from p.mov(Reg("eax"), Addr("x"))
            yield from p.mov(Addr("y"), 1)

        def t2(p: Processor) -> SimThread:
            yield from p.mov(Reg("ebx"), Addr("y"))
            yield from p.mov(Reg("ecx"), Addr("x"))

        def snapshot() -> Snapshot:
            return (
                p1.regs[Reg("eax")],
                p2.regs[Reg("ebx")],
                p2.regs[Reg("ecx")],
            )

        return tso, [t0, t1, t2], snapshot

    @staticmethod
    def target() -> Target:
        return (("p1.eax", 1), ("p2.ebx", 1), ("p2.ecx", 0)), False


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

        def t0(p: Processor) -> SimThread:
            p.regs[Reg("eax")] = 1  # initialize
            yield from p.xchg(Addr("x"), Reg("eax"))
            yield from p.mov(Reg("ebx"), Addr("y"))

        def t1(p: Processor) -> SimThread:
            p.regs[Reg("ecx")] = 1  # initialize
            yield from p.xchg(Addr("y"), Reg("ecx"))
            yield from p.mov(Reg("edx"), Addr("x"))

        def snapshot() -> Snapshot:
            return (
                p0.regs[Reg("ebx")],
                p1.regs[Reg("edx")],
            )

        return tso, [t0, t1], snapshot

    @staticmethod
    def target() -> Target:
        return (("p0.ebx", 0), ("p1.edx", 0)), False


class Ex_8_10_Demo:
    """Example 8-0.

    Stores are not reordered with locks.

    ---------------------------------
    P0              | P1
    ----------------+----------------
    XCHG [x] <- EAX | MOV EBX <- [y]
    MOV  [y] <- 1   | MOV ECX <- [x]
    ---------------------------------
    Initial state: P0:EAX=1
    Forbidden Final State: P1:EBX=1 and P1:ECX=0
    """

    @staticmethod
    def configure() -> Config:
        tso = TSO(nr_threads=2)
        _, p1 = tso.procs

        def t0(p: Processor) -> SimThread:
            p.regs[Reg("eax")] = 1  # initialize
            yield from p.xchg(Addr("x"), Reg("eax"))
            yield from p.mov(Addr("y"), 1)

        def t1(p: Processor) -> SimThread:
            yield from p.mov(Reg("ebx"), Addr("y"))
            yield from p.mov(Reg("ecx"), Addr("x"))

        def snapshot() -> Snapshot:
            return (
                p1.regs[Reg("ebx")],
                p1.regs[Reg("ecx")],
            )

        return tso, [t0, t1], snapshot

    @staticmethod
    def target() -> Target:
        return (("p1.ebx", 1), ("p1.ecx", 0)), False


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

        def t0(p: Processor) -> SimThread:
            yield from p.mov(Addr("x"), 1)
            yield from p.mfence()
            yield from p.mov(Reg("eax"), Addr("y"))

        def t1(p: Processor) -> SimThread:
            yield from p.mov(Addr("y"), 1)
            yield from p.mfence()
            yield from p.mov(Reg("ebx"), Addr("x"))

        def snapshot() -> Snapshot:
            return (
                p0.regs[Reg("eax")],
                p1.regs[Reg("ebx")],
            )

        return tso, [t0, t1], snapshot

    @staticmethod
    def target() -> Target:
        """The snapshot to check after the execution."""
        return (("p0.eax", 0), ("p1.ebx", 0)), False


@dataclass
class LinuxSpinlock:
    """Linux spinlock implementation discussed in the paper."""

    addr: Addr

    def lock(self, p: Processor) -> SimThread:
        # acquire loop
        while True:
            ret = Cell(0)
            yield from p.lock_dec(self.addr, ret)
            if ret.val >= 1:
                # was unlocked
                return

            # spin loop
            while True:
                yield from p.mov(Reg("ebx"), self.addr)
                if p.regs[Reg("ebx")] > 0:
                    # looks released, try to acquire again
                    break

    def unlock(self, p: Processor) -> SimThread:
        # ordinary move instead of lock;mov
        yield from p.mov(self.addr, 1)


class LinuxSpinlockDemo:
    """Linux (v2.6.24.7) spinlock demo from the paper.

    On entry the address of spinlock is in register EAX and the spinlock is
    unlocked iff its value is 1.

    acquire: LOCK DEC [EAX] ; LOCK’d decrement of [EAX]
             JNS enter      ; branch if [EAX] was >= 1
    spin:    CMP [EAX], 0   ; test [EAX]
             JLE spin       ; branch if [EAX] was <= 0
             JMP acquire    ; try again
    enter:                  ; the critical section starts here
    release: MOV [EAX] <- 1
    """

    @staticmethod
    def configure() -> Config:
        spinaddr = Addr("spinlock")  # use global address for spinlock
        tso = TSO(nr_threads=2, defmem={spinaddr: 1})
        spin = LinuxSpinlock(spinaddr)

        def t(p: Processor) -> SimThread:
            yield from spin.lock(p)
            yield from p.inc(Addr("counter"))
            yield from spin.unlock(p)

        def snapshot() -> Snapshot:
            return (tso.mem[Addr("counter")],)

        return tso, [t, t], snapshot

    @staticmethod
    def target() -> Target:
        return (("counter", 1),), False


@dataclass
class LinuxTicketedSpinlock:
    """Linux ticketed spinlock implementation discussed in the paper."""

    loaddr: Addr
    hiaddr: Addr

    def lock(self, p: Processor) -> SimThread:
        yield from p.mov(Reg("ecx"), 1)
        yield from p.lock_xadd(self.loaddr, Reg("ecx"))

        # spin loop
        while True:
            yield from p.mov(Reg("eax"), self.hiaddr)
            if p.regs[Reg("eax")] == p.regs[Reg("ecx")]:
                # now it is our ticket, released
                return

    def unlock(self, p: Processor) -> SimThread:
        # ordinary move instead of lock;mov
        yield from p.inc(self.hiaddr)


class LinuxTicketedSpinlockDemo:
    """Linux (v2.6.31) ticketed spinlock demo from the paper.

    acquire: MOV ECX <- 1           ; tkt := 1
             LOCK XADD [EBX] <- ECX ; atomic (tkt := [y]
                                    ;         [y] := tkt + 1
                                    ;         flush local write buffer)
    spin:    CMP [EAX], ECX         ; flag := ([x] = tkt)
             JE enter               ; if flag then goto enter
             JMP spin               ; goto spin
    enter:                          ; the critical section starts here
    release:                        ; [x] := [x] + 1
    """

    @staticmethod
    def configure() -> Config:
        spin = LinuxTicketedSpinlock(Addr("spinlo"), Addr("spinhi"))
        tso = TSO(nr_threads=2)

        def t(p: Processor) -> SimThread:
            yield from spin.lock(p)
            yield from p.inc(Addr("counter"))
            yield from spin.unlock(p)

        def snapshot() -> Snapshot:
            return (tso.mem[Addr("counter")],)

        return tso, [t, t], snapshot

    @staticmethod
    def target() -> Target:
        return (("counter", 1),), False


@dataclass
class JVMParker:
    """Skeleton of Parker class from JVM."""

    counter: Addr
    mutex: Mutex
    cond: Addr

    def _pthread_mutex_lock(self) -> SimThread:
        yield from self.mutex.lock()

    def _pthread_mutex_trylock(self, succ: Cell[bool]) -> SimThread:
        yield from schedule()
        if not self.mutex.locked:
            self.mutex.locked = True
            succ.val = True
        else:
            succ.val = False

    def _pthread_mutex_unlock(self, p: Processor) -> SimThread:
        # emulate release barrier with mfence
        yield from p.mfence()
        yield from self.mutex.unlock()

    def _pthread_cond_wait(self, p: Processor) -> SimThread:
        # do all this atomically
        self.mutex.locked = False

        p.mem[self.cond] = 1  # ready to get a signal
        yield from cond_schedule(
            lambda: not self.mutex.locked and p.mem[self.cond] == 2
        )

        p.mem[self.cond] = 0
        self.mutex.locked = True

    def _pthread_cond_signal(self, p: Processor) -> SimThread:
        yield from schedule()
        # emulate signal send as instant write to some addr
        if p.mem[self.cond] == 1:
            # if ready, deliver the signal, lost otherwise
            p.mem[self.cond] = 2

        yield from schedule()

    def park(self, p: Processor, *, bugged: bool) -> SimThread:
        yield from p.mov(Reg("counter"), self.counter)
        if p.regs[Reg("counter")] > 0:
            # fastpath
            yield from p.mov(self.counter, 0)
            if not bugged:
                # buggy version is missing mfence here
                yield from p.mfence()

            return

        succ = Cell(False)
        yield from self._pthread_mutex_trylock(succ)
        if not succ.val:
            return

        yield from p.mov(Reg("counter"), self.counter)
        if p.regs[Reg("counter")] > 0:
            # no wait needed
            yield from p.mov(self.counter, 0)
            yield from self._pthread_mutex_unlock(p)
            return

        yield from self._pthread_cond_wait(p)
        yield from p.mov(self.counter, 0)
        yield from self._pthread_mutex_unlock(p)

    def unpark(self, p: Processor) -> SimThread:
        yield from self._pthread_mutex_lock()
        yield from p.mov(Reg("unparkcounter"), self.counter)
        yield from p.mov(self.counter, 1)
        yield from self._pthread_mutex_unlock(p)
        if p.regs[Reg("unparkcounter")] < 1:
            yield from self._pthread_cond_signal(p)


class JVMParkerBugDemo:
    """HotSpot JVM Parker bug.

    See the ecoop10 paper for the full explanation.
    https://www.cl.cam.ac.uk/~pes20/weakmemory/ecoop10.pdf

    This implementation lacks any sync instruction in Parker::park fastpath and
    thus have a triangular race (see the paper). It leads to a lost
    Parker::unpark signal and thread hangup.
    """

    @staticmethod
    def configure(bugged: bool = True) -> Config:
        tso = TSO(nr_threads=2)
        pk = JVMParker(Addr("counter"), Mutex(), Addr("condvar"))

        def waiter(p: Processor):
            # init local var
            p.regs[Reg("wakeup")] = 0

            while True:
                yield from p.mov(Reg("eax"), Addr("sh1"))
                if p.regs[Reg("eax")] == 1:
                    break

                yield from pk.park(p, bugged=bugged)

            p.regs[Reg("wakeup")] = 1  # may lose signal and not get here

        def provider(p: Processor):
            # prepare the initial state, internal counter should be 1
            yield from pk.unpark(p)

            # reproduce lost wakeup
            yield from p.mov(Addr("sh1"), 1)
            yield from p.mfence()
            yield from pk.unpark(p)

        def snapshot() -> Snapshot:
            return (tso.procs[0].regs[Reg("wakeup")],)

        return tso, [waiter, provider], snapshot

    @staticmethod
    def target() -> Target:
        return (("wakeup", 0),), True


class JVMParkerBugFixedDemo:
    """HotSpot JVM Parker bug (fixed).

    See the ecoop10 paper for the full explanation.
    https://www.cl.cam.ac.uk/~pes20/weakmemory/ecoop10.pdf

    This implementation has the `mfence` instruction in fastpath so it does not
    suffer from lost signals as the original one.
    """

    @staticmethod
    def configure() -> Config:
        return JVMParkerBugDemo.configure(bugged=False)

    @staticmethod
    def target() -> Target:
        return (("break1", 0),), False


class PetersonMutex:
    """Classic example of concurrent mutex algorithm.

    https://en.wikipedia.org/wiki/Peterson's_algorithm
    """

    @staticmethod
    def lock(tid: int, p: Processor, use_mb: bool) -> SimThread:
        me = tid
        other = 1 - tid
        yield from p.mov(Addr(f"f{me}"), 1)
        yield from p.mov(Addr("turn"), other)
        if use_mb:
            yield from p.mfence()

        while True:
            yield from p.mov(Reg("eax"), Addr(f"f{other}"))
            yield from p.mov(Reg("ebx"), Addr("turn"))
            if not (p.regs[Reg("eax")] and p.regs[Reg("ebx")] == other):
                return

    @staticmethod
    def unlock(tid: int, p: Processor) -> SimThread:
        me = tid
        yield from p.mov(Addr(f"f{me}"), 0)


class PetersonLockDemo:
    """Peterson's locking algorithm example.

    This demo shows that naive implementation of Peterson's algorithm is broken
    on x86* due to store buffering (non-sequentially consistent memory writes).

    --------------------------------------------------------
    P0                           | P1
    -----------------------------+--------------------------
    ; lock                       | ; lock
    MOV [f0]   <- True           | MOV [f1]   <- True
    MOV [turn] <- 1              | MOV [turn] <- 0
    while True:                  | while True:
        MOV EAX <- [f1]          |   MOV EAX <- [f0]
        MOV EBX <- [turn]        |   MOV EBX <- [turn]
        if not (EAX and EBX==1): |   if not (EAX and EBX==0):
          break                  |     break
                                 |
    ; critsect                   | ; critsect
    INC [counter]                | INC [counter]
                                 |
    ; unlock                     | ; unlock
    MOV [f0] <- False            | MOV [f1] <- False
    -----------------------------------------------------
    Allowed Final State: [counter]=1.

    x86-TSO explanation:
    - all locking stores are buffered in both P0 and P1
    - P0 reads [f1] from memory (gets 0)
    - P1 reads [f0] from memory (gets 0)
    - (at any time after this point the buffs are flushed, but it is too late)
    - both conditions are true, break the loop
    - both threads are in critical section, data race over [counter]
    """

    @staticmethod
    def configure() -> Config:
        tso = TSO(nr_threads=2)

        def t0(p: Processor) -> SimThread:
            yield from PetersonMutex.lock(0, p, use_mb=False)
            yield from p.inc(Addr("counter"))
            yield from PetersonMutex.unlock(0, p)

        def t1(p: Processor) -> SimThread:
            yield from PetersonMutex.lock(1, p, use_mb=False)
            yield from p.inc(Addr("counter"))
            yield from PetersonMutex.unlock(1, p)

        def snapshot() -> Snapshot:
            return (tso.mem[Addr("counter")],)

        return tso, [t0, t1], snapshot

    @staticmethod
    def target() -> Target:
        """The snapshot to check after the execution."""
        return (("[counter]", 1),), True


class PetersonLockFixedDemo:
    """Peterson's locking algorithm example (fixed).

    This demo shows that inserting `mfence` on x86 restores sequential
    consistency for store operations, making Peterson's algorithm correct.

    -----------------------------------------------------------
    P0                           | P1
    -----------------------------+-----------------------------
    ; lock                       | ; lock
    MOV [f0]   <- True           | MOV [f1]   <- True
    MOV [turn] <- 1              | MOV [turn] <- 0
    MFENCE ; <<<============ FIX | MFENCE ; <<<============ FIX
    while True:                  | while True:
        MOV EAX <- [f1]          |     MOV EAX <- [f0]
        MOV EBX <- [turn]        |     MOV EBX <- [turn]
        if not (EAX and EBX==1): |     if not (EAX and EBX==0):
          break                  |         break
                                 |
    ; critsect                   | ; critsect
    INC [counter]                | INC [counter]
                                 |
    ; unlock                     | ; unlock
    MOV [f0] <- False            | MOV [f1] <- False
    ------------------------------------------------------------
    Forbidden Final State: [counter]=1.
    """

    @staticmethod
    def configure() -> Config:
        tso = TSO(nr_threads=2)

        def t0(p: Processor) -> SimThread:
            yield from PetersonMutex.lock(0, p, use_mb=True)
            yield from p.inc(Addr("counter"))
            yield from PetersonMutex.unlock(0, p)

        def t1(p: Processor) -> SimThread:
            yield from PetersonMutex.lock(1, p, use_mb=True)
            yield from p.inc(Addr("counter"))
            yield from PetersonMutex.unlock(1, p)

        def snapshot() -> Snapshot:
            return (tso.mem[Addr("counter")],)

        return tso, [t0, t1], snapshot

    @staticmethod
    def target() -> Target:
        """The snapshot to check after the execution."""
        return (("[counter]", 1),), False


def play_demo(demo: type[Demo], *, iters: int = 0) -> bool:
    """Play the demo from the template."""
    # prepare the demo
    tso, thrctrs, snapshot = demo.configure()

    assert len(tso.procs) == len(thrctrs), "must be exact amount of threads"

    print(demo.__doc__)
    print("Running the simulator, hit Ctrl+C to stop...")

    # run the demo and collect the observed states
    outputs = ObservedStats()
    simsched(
        itertools.chain(
            [partial(p.wrap, t) for p, t in zip(tso.procs, thrctrs)],
            tso.sbctrs,
        ),
        loopctr=partial(reg_count_looper, outputs, snapshot, tso, iters),
    )

    print("interrupted\n")
    print(f"Observed states (total {len(outputs.storage)}):")

    # pretty print the outputs
    target, allowed = demo.target()
    target_name = [name for name, _ in target]
    target_snap = tuple(snap for _, snap in target)
    print(outputs.describe(target_name))

    # investigate the target snapshot
    status = "Allowed" if allowed else "Forbidden"
    count = outputs.storage.get(target_snap, 0)
    print(f"{status} state: {target} happened {count} times")

    return allowed == bool(count)


DEMOS: Mapping[str, type[Demo]] = {
    # demos from the original papers
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
    "ex8-10": Ex_8_10_Demo,
    "amd5": Amd5Demo,
    "linux-spin": LinuxSpinlockDemo,
    "linux-spin-ticket": LinuxTicketedSpinlockDemo,
    "jvm-parker": JVMParkerBugDemo,
    "jvm-parker-fix": JVMParkerBugFixedDemo,
    # other demos
    "peterson": PetersonLockDemo,
    "peterson-fix": PetersonLockFixedDemo,
}


def test() -> bool:
    """Helper function to run all tests for debug."""
    for name, demo in DEMOS.items():
        io = StringIO()
        print(name, end=": ")
        with contextlib.redirect_stdout(io):
            result = play_demo(demo, iters=10000)

        if result:
            print("OK")
        else:
            print("FAIL")
            print(io.getvalue())
            return False

    return True


def demos_list() -> str:
    """Return human readable string with all demos."""
    return "(\n    " + ",\n    ".join(DEMOS) + ",\n)"


def main() -> None:
    """CLI entrypoint."""
    prog, *args = sys.argv

    def usage() -> None:
        print(f"usage: {prog} {demos_list()}")

    match args:
        case ["--test"]:
            success = test()
        case [name]:
            if name not in DEMOS:
                usage()
                sys.exit(1)

            success = play_demo(DEMOS[name])
        case _:
            usage()
            sys.exit(1)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
