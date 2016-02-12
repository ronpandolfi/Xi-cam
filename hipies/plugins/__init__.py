from collections import OrderedDict


modules = []
plugins = OrderedDict()


def loadplugins(placeholders):
    import viewer, timeline, library, fluctuationscattering, ipythonconsole

    global plugins, modules
    modules = [viewer, timeline, library, ipythonconsole, fluctuationscattering]

    for module in modules:
        plugin = module.plugin(placeholders)
        module.plugininstance = plugin
        plugins[plugin.name] = plugin