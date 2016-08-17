import distributed
from tornado.ioloop import IOLoop
from threading import Thread
from distributed import Scheduler, Worker, Executor

import paramiko
import select

import getpass
import os
import socket
import select
import threading
from pipeline import msg

try:
    import SocketServer
except ImportError:
    import socketserver as SocketServer

g_verbose = True


def verbose(s):
    if g_verbose:
        msg.logMessage(s,msg.DEBUG)


class ForwardServer(SocketServer.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


class Handler(SocketServer.BaseRequestHandler):
    def handle(self):
        try:
            chan = self.ssh_transport.open_channel('direct-tcpip',
                                                   (self.chain_host, self.chain_port),
                                                   self.request.getpeername())
        except Exception as e:
            verbose('Incoming request to %s:%d failed: %s' % (self.chain_host,
                                                              self.chain_port,
                                                              repr(e)))
            return
        if chan is None:
            verbose('Incoming request to %s:%d was rejected by the SSH server.' %
                    (self.chain_host, self.chain_port))
            return

        verbose('Connected!  Tunnel open %r -> %r -> %r' % (self.request.getpeername(),
                                                            chan.getpeername(), (self.chain_host, self.chain_port)))
        while True:
            r, w, x = select.select([self.request, chan], [], [])
            if self.request in r:
                data = self.request.recv(1024)
                if len(data) == 0:
                    break
                chan.send(data)
            if chan in r:
                data = chan.recv(1024)
                if len(data) == 0:
                    break
                self.request.send(data)

        peername = self.request.getpeername()
        chan.close()
        self.request.close()
        verbose('Tunnel closed from %r' % (peername,))


class DaskScheduler(object):
    def __init__(self, client, ipaddr, port, password, runscript):
        self.client = client
        self.ipaddr = ipaddr
        self.port = port
        self.runscript = runscript
        self.password = password
        # self.command = "/usr/common/graphics/visit/camera/runscript.sh {0} {1}".format(ipaddr, port)
        self.command = runscript + " {0} {1}".format(ipaddr, port)
        msg.logMessage(self.command,msg.DEBUG)

    def serve(self):
        msg.logMessage("Serving: " + self.command,msg.INFO)
        channel = self.client.get_transport().open_session()
        msg.logMessage(channel.get_pty(),msg.DEBUG)
        # channel.exec_command('tty')
        channel.exec_command(self.command)
        readytorun=False
        while True:
            if channel.exit_status_ready():
                break
            rl, wl, xl = select.select([channel], [], [], 10.0)
            if len(rl) > 0:
                line = channel.recv(1024)
                msg.logMessage(line)
                if 'queued and waiting for resources' in line:
                    msg.showMessage('Execution queued and waiting for resources...')
                elif 'has been allocated resources' in line:
                    msg.showMessage('Execution has been allocated resources...')
                    readytorun=True
                if readytorun and line:
                    msg.showMessage('Executing job...')
                    readytorun=False

                if line.find("Password") >= 0:
                    msg.logMessage("writing password",msg.DEBUG)
                    channel.sendall(self.password + "\n")
        msg.logMessage("Ending Dask Scheduler",msg.INFO)


class DaskWorker(object):
    def __init__(self, client, ipaddr, port, nodes, partition, time, procs, threads):
        self.client = client
        self.ipaddr = ipaddr
        self.port = port
        self.nodes = nodes
        self.partition = partition
        self.time = time
        self.procs = procs
        self.threads = threads

        self.command = "/usr/common/graphics/visit/camera/runlocalserver.sh {0} {1}".format(ipaddr, port)
        # self.command = "/usr/common/graphics/visit/camera/runserver.sh {0} {1} {2} {3} {4} {5} {6}".format(ipaddr, port, nodes, partition, time, procs, threads)
        msg.logMessage(self.command,msg.DEBUG)

    def serve(self):
        msg.logMessage("Serving Worker: " + self.command,msg.INFO)
        channel = self.client.get_transport().open_session()
        channel.exec_command(self.command)
        # client.exec_command(self.command)
        while True:
            # if channel.exit_status_ready():
            #  break
            rl, wl, xl = select.select([channel], [], [], 10.0)
            if len(rl) > 0:
                msg.logMessage("rl:", channel.recv(1024),msg.DEBUG)
        msg.logMessage("Ending Dask Worker",msg.INFO)


# client = paramiko.SSHClient()
# client.load_system_host_keys()
# client.connect('edison.nersc.gov', username="hkrishna")

class RemoteScheduler():
    """
       Create a remote executor
    """

    def __init__(self, addr, username, loop, password, machine, runscript):
        self.loop = loop
        self.addr = addr
        self.username = username
        self.password = password
        self.command = runscript
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.load_system_host_keys()
        if len(password) > 0:
            self.client.connect(addr, username=username, password=password)
        else:
            self.client.connect(addr, username=username)
        stdin, stdout, stderr = self.client.exec_command(
            "python -c 'import socket; s=socket.socket(); s.bind((\"\", 0)); result = socket.gethostbyname(socket.gethostname()) + \":\" + str(s.getsockname()[1]); s.close(); print(result)'")
        stdin.close()

        for line in stdout.read().splitlines():
            result = line

        self.remote_addr = result.split(":")
        self.local_port = self.get_free_port()
        if len(machine) > 0:
            self.remote_addr[0] = machine

        msg.logMessage((self.local_port, self.remote_addr),msg.DEBUG)
        self.start_scheduler(self.remote_addr[0], self.remote_addr[1], self.client, password, runscript)
        self.executor = None
        self.forward_tunnel(self.local_port, self.remote_addr[0], self.remote_addr[1], self.client.get_transport())

    def execute(self):
        msg.logMessage("Starting Executor",msg.INFO)
        # self.executor = Executor("{0}:{1}".format(self.remote_addr[0], self.remote_addr[1]))
        self.executor = Executor("{0}:{1}".format("localhost", self.local_port))
        msg.logMessage("End Executor",msg.INFO)

    def close(self):
        self.client.exec_command("killall dask-scheduler dask-worker")
        self.client.exec_command("killall " + os.path.basename(self.command))

    def get_free_port(self):
        s = socket.socket()
        s.bind(("", 0))
        localport = s.getsockname()[1]
        s.close()
        return localport

    """
    def start_slurm_worker(self, ipaddr, port, nodes, partition, time, procs, threads, client):
       self.dask_worker = DaskWorker(client, ipaddr, port, nodes, partition, time, procs, threads)
       server_thread = threading.Thread(target=dask_worker.serve)
       server_thread.daemon = True
       server_thread.start()
    """

    def start_scheduler(self, ipaddr, port, client, password, runscript):
        self.dask_sched = DaskScheduler(client, ipaddr, port, password, runscript)
        server_thread = threading.Thread(target=self.dask_sched.serve)
        # server_thread.daemon = True
        server_thread.start()

    def forward_tunnel(self, local_port, remote_host, remote_port, transport):
        class SubHander(Handler):
            chain_host = remote_host
            chain_port = int(remote_port)
            ssh_transport = transport

        fs = ForwardServer(('', int(local_port)), SubHander)
        msg.logMessage(("FORWARDING: ", local_port, remote_host, remote_port),msg.INFO)
        server_thread = threading.Thread(target=fs.serve_forever)
        server_thread.daemon = True
        server_thread.start()
