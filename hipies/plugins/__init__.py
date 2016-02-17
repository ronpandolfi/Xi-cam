from collections import OrderedDict


modules = []
plugins = OrderedDict()


def initplugins(placeholders):
    import viewer, timeline, library, fluctuationscattering, ipythonconsole

    global plugins, modules
    modules = [viewer, timeline, library, ipythonconsole, fluctuationscattering]

    for module in modules:
        link = pluginlink(module.plugin, placeholders)
        link.enable()
        plugins[link.instance.name] = link


class pluginlink():
    def __init__(self, plugin, placeholders):
        self.plugin = plugin
        self.instance = None
        self.placeholders = placeholders

    def disable(self):
        self.instance = None

    def enable(self):
        self.instance = self.plugin(self.placeholders)

    @property
    def enabled(self):
        return self.instance is not None