from pipeline import msg

packages = {}
try:
    import tomopy
    packages['tomopy'] = tomopy
    msg.logMessage('tomopy module loaded')
except ImportError:
    msg.logMessage('tomopy module not available', level=30)  # 30 -> warning
    packages['tomopy'] = None
    tomopy = None
try:
    import astra
    packages['astra'] = astra
    msg.logMessage('Astra module loaded')
except ImportError:
    msg.logMessage('astra module not available', level=30)
    packages['astra'] = None
    astra = None