import warnings

packages = {}
try:
    import tomopy
    packages['tomopy'] = tomopy
    print 'tomopy module loaded'
except ImportError:
    warnings.warn('tomopy module not available')
    packages['tomopy'] = None
try:
    import astra
    packages['astra'] = astra
    print 'astra module loaded'
except ImportError:
    warnings.warn('astra module not available')
    packages['astra'] = None