import traceback
import importlib
from pipeline import msg
from PySide import QtGui

def import_module(modname,packagename=None):
    module=None
    try:
        module=importlib.import_module(modname, packagename)
        msg.logMessage(("Imported", modname), msg.DEBUG)
    except ImportError as ex:
        msg.logMessage('Module could not be loaded: ' + modname)
        tb = traceback.format_exc()
        msg.logMessage('ImportError message: ' + ex.message)
        msg.logMessage(tb)

        missingpackage = ex.message.replace('No module named ', '')

        import config
        if config.settings['ignoredmodules']:
            if missingpackage in config.settings['ignoredmodules']:
                return None

        msgBox = QtGui.QMessageBox()
        msgBox.setText("A python package is missing! Xi-cam can try to install this for you.")
        msgBox.setInformativeText("Would you like to install " + missingpackage + "?")
        msgBox.setStandardButtons(QtGui.QMessageBox.Yes | QtGui.QMessageBox.No | QtGui.QMessageBox.Ignore)
        msgBox.setDefaultButton(QtGui.QMessageBox.Yes)

        response = msgBox.exec_()

        if response == QtGui.QMessageBox.Yes:
            import pip

            failure=pip.main(['install', '--user', missingpackage])
            if failure:
                failure=pip.main(['install', missingpackage])

            if not failure:
                msgBox = QtGui.QMessageBox()
                msgBox.setText('Success! The missing package, ' + missingpackage + ', has been installed!')
                msgBox.setInformativeText('Please restart Xi-cam now.')
                msgBox.setStandardButtons(QtGui.QMessageBox.Ok)
                msgBox.exec_()
                exit(0)
            else:
                if modname.strip('.') == 'MOTD':
                    from xicam import debugtools
                    debugtools.frustration()
                    msgBox = QtGui.QMessageBox()
                    msgBox.setText(
                        'Sorry, ' + missingpackage + ' could not be installed. This is a Xi-cam critical library.')
                    msgBox.setInformativeText('Xi-cam cannot be loaded . Please install ' + missingpackage + ' manually.')
                    msgBox.setStandardButtons(QtGui.QMessageBox.Ok)
                    msgBox.exec_()
                    exit(1)
                else:
                    from xicam import debugtools
                    debugtools.frustration()
                    msgBox = QtGui.QMessageBox()
                    msgBox.setText(
                        'Sorry, ' + missingpackage + ' could not be installed. Try installing this package yourself, or contact the package developer.')
                    msgBox.setInformativeText('Would you like to continue loading Xi-cam?')
                    msgBox.setStandardButtons(QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
                    response = msgBox.exec_()
                    if response == QtGui.QMessageBox.No:
                        exit(1)
        elif response == QtGui.QMessageBox.Ignore and modname.strip('.') != 'MOTD':
            import config
            if config.settings['ignoredmodules']:
                config.settings['ignoredmodules'].append(missingpackage)
            else:
                config.settings['ignoredmodules']=[missingpackage]
            msgBox = QtGui.QMessageBox()
            msgBox.setText('Xi-cam will no longer prompt you to install this package, however some plugins may be disabled.')
            msgBox.setStandardButtons(QtGui.QMessageBox.Ok)
            msgBox.exec_()

        if modname.strip('.') == 'MOTD':
            from xicam import debugtools
            debugtools.frustration()
            msgBox = QtGui.QMessageBox()
            msgBox.setText(
                'Sorry, ' + missingpackage + ' is a Xi-cam critical library. This must be installed to run Xi-cam!')
            msgBox.setInformativeText('Xi-cam cannot be loaded . Please install ' + modname.strip('.') + ' manually.')
            msgBox.setStandardButtons(QtGui.QMessageBox.Ok)
            msgBox.exec_()
            exit(1)


    if not module: msg.logMessage('Failed to import '+modname,msg.CRITICAL)
    return module
