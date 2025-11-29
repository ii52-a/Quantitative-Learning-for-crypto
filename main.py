import sys

from PyQt5.QtWidgets import QApplication

import UI.Ui as Ui
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Ui.GetbackUi()
    window.show()
    sys.exit(app.exec_())