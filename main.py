import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
import GUIrun




if __name__ == '__main__':
    app = QApplication(sys.argv)

    mainW = QMainWindow()
    ui = GUIrun.Ui_MainWindow()
    ui.setupUi(mainW)

    mainW.show()
    sys.exit(app.exec_())

