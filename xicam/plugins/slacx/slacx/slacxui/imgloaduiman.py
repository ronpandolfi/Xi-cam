import os
from functools import partial

from PySide import QtCore, QtGui, QtUiTools

from ..slacxcore import slacximg 
from ..slacxcore import slacxex

class ImgLoadUiManager(object):

    def __init__(self,ui_file,imgman,imgfile):
        ui_file.open(QtCore.QFile.ReadOnly)
        self.ui = QtUiTools.QUiLoader().load(ui_file)
        ui_file.close()
        self.imgman = imgman 
        self.imgfile = imgfile 
        self.setup_ui()

    def setup_ui(self):
        """Open a UI to request a tag for image before loading from imgfile"""
        self.ui.prompt_box.setPlainText(
        'Enter a unique tag for: \n{}'.format(self.imgfile))
        self.ui.prompt_box.setMaximumHeight(200)
        self.ui.prompt_box.setReadOnly(True)
        self.ui.tag_entry.setText(self.default_tag())
        self.ui.finish_button.setText('Finish')
        # Set button to activate on Enter key?
        self.ui.finish_button.setDefault(True)
        self.ui.tag_entry.returnPressed.connect(partial(self.load_img_with_tag,self.imgfile))
        self.ui.finish_button.clicked.connect(partial(self.load_img_with_tag,self.imgfile))
    
    def load_img_with_tag(self,imgfile):
        # Check the tag. Proceed only if it's a good tag.
        tag = self.ui.tag_entry.text()
        result = self.imgman.is_good_tag(tag)
        if result[0]:
            ins_row = self.imgman.rowCount(QtCore.QModelIndex())
            new_img = slacximg.SlacxImage(imgfile)
            # Add this SlacxImage to ImgManager tree, self.imgman
            self.imgman.add_image(new_img,tag)
            indx = self.imgman.index(ins_row,0,QtCore.QModelIndex())
            # Add image pixel data  
            new_img.load_img_data()
            self.imgman.add_image_data(indx,new_img.img_data,'img_data',new_img.size_tag())
            #self.ui.image_tree.setCurrentIndex(
            #self.imgman.index(ins_row,0,QtCore.QModelIndex()))
            self.ui.close()
        else:
            # Request a different tag
            msg_ui = slacxex.start_message_ui()
            msg_ui.setParent(self.ui,QtCore.Qt.Window)
            msg_ui.setWindowTitle("Tag Error")
            msg_ui.message_box.setPlainText(
            'Tag error for {}: \n{} \n\n'.format(tag, result[1])
            + 'Enter a unique alphanumeric tag, '
            + 'using only letters, numbers, -, and _. ')
            # Set button to activate on Enter key
            msg_ui.ok_button.setFocus()
            msg_ui.show()

    def default_tag(self):
        indx = 0
        goodtag = False
        while not goodtag:
            testtag = 'img{}'.format(indx)
            if not testtag in self.imgman.list_tags(QtCore.QModelIndex()):
                goodtag = True
            else:
                indx += 1
        return testtag

