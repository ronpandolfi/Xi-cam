from collections import OrderedDict


modules = []
plugins = OrderedDict()


def loadplugins(placeholders):
    import viewer, timeline, library, fluctuationscattering, ipythonconsole, spoth5file

    global plugins, modules
    modules = [viewer, timeline, library, ipythonconsole, fluctuationscattering, spoth5file]

    for module in modules:
        plugin = module.plugin(placeholders)
        module.plugininstance = plugin
        plugins[plugin.name] = plugin
