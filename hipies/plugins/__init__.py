__all__ = ['viewer', 'widgets', 'timeline', 'library', 'base']

import viewer, timeline, library
from collections import OrderedDict

pluginclasses = [viewer.plugin, timeline.plugin, library.plugin]
plugins = OrderedDict()

def loadplugins(placeholders):
    global plugins
    for plugin in pluginclasses:
        plugin = plugin(placeholders)
        plugins[plugin.name] = plugin