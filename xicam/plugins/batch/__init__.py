from __future__ import absolute_import
from __future__ import unicode_literals

from .. import base

from pipeline import msg
from paws.ui.widgets import BatchWidget
from paws.api import PawsAPI

class BatchPlugin(base.plugin):
    name = 'Batch'

    def __init__(self, *args, **kwargs):

        self.paw = PawsAPI()
        self.pawswidget = BatchWidget.BatchWidget(self.paw)
        self.centerwidget = self.pawswidget.ui.viewer_box
        self.rightwidget = self.pawswidget.ui.wf_box
        self.bottomwidget = self.pawswidget.ui.batch_box
        self.batchlist = self.pawswidget.ui.batch_list


        super(BatchPlugin, self).__init__(*args, **kwargs)

    def openfiles(self, files, operation=None, operationname=None):
        self.batchlist.addItems(files)