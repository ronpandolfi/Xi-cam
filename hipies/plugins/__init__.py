__all__ = ['viewer', 'widgets', 'timeline', 'library', 'base', 'fluctuationscattering']

import viewer, timeline, library, fluctuationscattering
from collections import OrderedDict

modules = [viewer, timeline, library]
plugins = OrderedDict()

def loadplugins(placeholders):
    global plugins
    for module in modules:
        plugin = module.plugin(placeholders)
        module.plugininstance = plugin
        plugins[plugin.name] = plugin