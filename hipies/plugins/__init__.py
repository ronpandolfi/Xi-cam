__all__ = ['viewer', 'widgets', 'timeline', 'library', 'base', 'fluctuationscattering']

import viewer, timeline, library, fluctuationscattering
from collections import OrderedDict

pluginclasses = [viewer.plugin, timeline.plugin, library.plugin, fluctuationscattering.plugin]
plugins = OrderedDict()

def loadplugins(placeholders):
    global plugins
    for plugin in pluginclasses:
        plugin = plugin(placeholders)
        plugins[plugin.name] = plugin