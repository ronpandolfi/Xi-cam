import distributed 
from tornado.ioloop import IOLoop
from threading import Thread
from distributed import Scheduler, Worker, Executor
import logging
import atexit

distributed.core.logging.propagate = False

__ioloop__ = IOLoop()

class DaskLoop():
  def __init__(self):
    self.loop = __ioloop__
    self.t = Thread(target=self.loop.start)
    self.t.start()


#def close_ioloop():
#    print "closing..."
#    __ioloop__.close()

#atexit.register(close_ioloop)
#print "registering.."
