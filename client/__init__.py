from __future__ import absolute_import
from __future__ import unicode_literals
import pysftp
import paramiko
from .globus import GLOBUSError
from .spot import SPOTError
from .newt import NEWTError
from . import ssh

__all__ = ['newt', 'spot', 'globus', 'sftp', 'ssh']

# Exceptions raised by clients that we care to handle
EXCEPTIONS = (pysftp.ConnectionException, paramiko.ssh_exception.BadAuthenticationType,
              paramiko.ssh_exception.AuthenticationException, GLOBUSError, SPOTError, GLOBUSError)

