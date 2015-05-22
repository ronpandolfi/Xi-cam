
import hipies
import sys

window = hipies.main.MyMainWindow()
window.ui.show()

sys.exit(window.app.exec_())