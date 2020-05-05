import sys

from PyQt5.QtWidgets import QApplication

from app.equilaizer import Main_Window_class

if __name__ == '__main__':
    Equalizer = QApplication(sys.argv)
    Main_Window = Main_Window_class()
    sys.exit(Equalizer.exec_())
