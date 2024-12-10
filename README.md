# simsched - simulated scheduling

`simsched` is a simple tool to troubleshoot/investigate concurrent algorithms.

The idea of the tool:

- You write a prototype of your algorithm using Python synchronous coroutines
  with explicit "traps" for interleaving. It should be easy thing to do: Python
  coroutines are powerful and easy to write and `simsched` package also provides
  some library support for common tasks (i.e. Mutex).
- You feed these coroutines to the simulator engine, which simulates the
  scheduler of the "tasks".
- The engine runs in an endless loop trying to reach as many paths of execution
  as possible using random scheduling.
- You tweak the simulation loop by tuning the LoopController object to achieve
  your goal: collect observed states or collect "bad"/"good" sequences or
  whatsoever.

## How to start

Install `simsched` as a usual `pip` package. Then go to `examples` directory and
try to run some of them to get the idea. The good examples to start with:

- `nonatomic_counter`: simulates integer counter being updated from different
  threads in non-atomic manner. This demo collects observed final values of the
  counter after all the threads are finished.
- `mutexes`: simulates 3 mutexes and 3 threads, each trying to lock two of the
  mutexes. The order of taking mutexes is incorrect, allowing circular deadlock.
  This demo collects traces leading to a deadlock situation.
