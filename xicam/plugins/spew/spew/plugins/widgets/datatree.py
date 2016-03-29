# -*- coding: utf-8 -*-
"""
@author: lbluque
"""

from PySide import QtGui
from pyqtgraph.parametertree import ParameterTree

class MetadataTree(ParameterTree):
    """
    Table with a ParamTree for viewing and editing file metadata
    """

    def __init__(self, params, parent=None, showHeader=True):
        super(MetadataTree, self).__init__(parent=parent, showHeader=showHeader)
        for param in params:
            self.addParameters(param)
