import importlib
from pipeline import msg
import tomopy

# Packages to import
PACKAGE_LIST = ['astra', 'dxchange']

# Dictionary with package names as keys and package objects as values
packages = {}

for name in PACKAGE_LIST:
    try:
        package = importlib.import_module(name)
        print package
        packages[name] = package
        msg.logMessage('{} module loaded'.format(name), level=20)
    except ImportError as ex:
        msg.logMessage('{} module not available'.format(name), level=30)  # 30 -> warning'

# Tomopy is actually necessary, so its important that its a hard import
packages['tomopy'] = tomopy

# Add the extra functions
import pipelinefunctions, mbir
packages['pipelinefunctions'] = pipelinefunctions
packages['mbir'] = mbir
