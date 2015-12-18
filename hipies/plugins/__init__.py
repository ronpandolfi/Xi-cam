__all__ = ['viewer', 'widgets', 'timeline', 'library', 'base', 'fluctuationscattering', 'ipythonnb']

import viewer, timeline, library, fluctuationscattering
from collections import OrderedDict
import ipythonnb

modules = [viewer, timeline, library, ipythonnb]
plugins = OrderedDict()


def loadplugins(placeholders):
    global plugins
    for module in modules:
        plugin = module.plugin(placeholders)
        module.plugininstance = plugin
        plugins[plugin.name] = plugin