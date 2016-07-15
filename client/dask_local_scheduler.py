import distributed 
from tornado.ioloop import IOLoop
from threading import Thread
from distributed import Scheduler, Worker, Executor
import logging

class LocalScheduler():
  
  def __init__(self, dloop):
    """
      Create a local scheduler and worker to run commands
    """
    self.dloop = dloop
    self.loop = dloop.loop

    print "Starting scheduler"
    print self.dloop
    print self.dloop.loop

    print "Starting scheduler"
    self.s = Scheduler(loop=self.dloop.loop)
    self.s.start(0) #random port
    print "End scheduler"

    print "Starting worker"
    self.w = Worker(self.s.ip, self.s.port, ncores=1, loop=self.loop)
    self.w.start(0)

    self.addr = self.s.ip
    self.port = self.s.port

  def execute(self):
    print "Starting executor"
    self.executor = Executor("{0}:{1}".format(self.addr,self.port))
    print "executor.."

  def close(self):
    self.s.close()
