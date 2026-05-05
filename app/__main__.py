# __main__.py
import sys
from PyQt6.QtWidgets import QApplication
#from .application.controller import AppController
from .ui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    #controller = AppController()
    
    window = MainWindow()
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()