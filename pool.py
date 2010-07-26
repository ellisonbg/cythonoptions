"""pool.py: A simple poll implementation using threads or processes.

The following looks like some sort of problem with multiprocessing::

    from scipy.special import lambertw
    from pool import SimpleProcessPool
    p = SimpleProcessPool(1)
    p.map(lambertw, range(100)) # OK
    p.map(lambertw, range(1000)) # hangs on out_queue.join()
"""

import multiprocessing
import Queue
from Queue import Empty
import threading

class SimpleThreadPool(object):
    """A simple thread pool."""

    def __init__(self, size=2):
        self.size = size

    def _do_work(self, f, in_queue, out_queue):
        while True:
            try:
                arg = in_queue.get(block=False)
                result = f(arg)
                out_queue.put(result)
                # This allows us to call in_queue.join().
                in_queue.task_done()
            except Empty:
                break

    def map(self, f, seq):
        """A parallel version of Python's map function."""

        assert callable(f), "first argument must be a single argument callable"
        in_queue = Queue.Queue()
        out_queue = Queue.Queue()
        for arg in seq:
            in_queue.put(arg)
        workers = [
            threading.Thread(
                target=self._do_work, 
                args=(f, in_queue, out_queue)
            ) for i in range(self.size)]
        for w in workers:
            w.start()
        in_queue.join()
        results = [out_queue.get(block=False) for n in range(out_queue.qsize())]
        for w in workers:
            w.join()
        return results

class SimpleProcessPool(object):
    """A simple process pool."""

    def __init__(self, size=2):
        self.size = size

    def _do_work(self, f, in_queue, out_queue):
        while True:
            try:
                arg = in_queue.get(block=False)
                result = f(arg)
                print "Did iteration for arg:", arg
                out_queue.put(result)
                # This allows us to call in_queue.join() below.
                in_queue.task_done()
            except Empty:
                break

    def map(self, f, seq):
        """A parallel version of Python's map function."""

        assert callable(f), "first argument must be a single argument callable"
        in_queue = multiprocessing.JoinableQueue()
        out_queue = multiprocessing.JoinableQueue()
        for arg in seq:
            in_queue.put(arg)
        workers = [
            multiprocessing.Process(
                target=self._do_work, 
                args=(f, in_queue, out_queue)
            ) for i in range(self.size)]
        for w in workers:
            w.start()
        in_queue.join()
        results = []
        while True:
            try:
                result = out_queue.get(block=False)
            except Empty:
                break
            else:
                results.append(result)
        for w in workers:
            w.join()

        return results

