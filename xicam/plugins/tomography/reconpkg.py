import importlib
from pipeline import msg

PACKAGE_LIST =['tomopy', 'astra', 'dxchange']
packages = {}

for name in PACKAGE_LIST:
    try:
        package = importlib.import_module(name)
        print package
        packages[name] = package
        msg.logMessage('{} module loaded'.format(name), level=20)
    except ImportError:
        msg.logMessage('{} module not available'.format(name), level=30)  # 30 -> warning

import pipelinefunctions
packages['pipelinefunctions'] = pipelinefunctions
