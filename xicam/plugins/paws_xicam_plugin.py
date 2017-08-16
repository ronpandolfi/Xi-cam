from . import base
import paws
from paws import api as pawsapi
from paws import ui as pawsui
from paws.ui.UiManager import UiManager

class PawsXicamPlugin(base.plugin):
    # The display name in the xi-cam plugin bar
    name = 'Workflow'

    def __init__(self, *args, **kwargs):
        app = pawsui.ui_app()
        # start core objects
        xipaw = pawsapi.start()
        # start a ui manager
        ui_manager = UiManager(xipaw,app)

        # Set the widgets in base.plugin containers
        self.centerwidget = ui_manager.ui
        self.leftwidget = None
        self.rightwidget = None
        self.bottomwidget = None

        # There seems to be a problem with this plugin loading approach,
        # where the frames sometimes fail 
        # to bring their children with them.
        # Adding calls to findChildren() 
        # seems to force the frames
        # to include their children.
        # -LAP
        ui_manager.ui.findChildren(object)

        super(PawsXicamPlugin, self).__init__(*args, **kwargs)

