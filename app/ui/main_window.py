from PyQt6.QtWidgets import (
    QMainWindow, QSplitter, QWidget,
    QGridLayout, QFrame, QLabel, QVBoxLayout, QGroupBox, QFormLayout, QDoubleSpinBox
)
from PyQt6.QtCore import Qt

class PlaceholderPlot(QFrame):
    def __init__(self, index):
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("background-color: #1e1e2e; border: 1px solid #444;")
        layout = QGridLayout(self)
        label = QLabel(f"Graph {index}")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("color: #555; font-size: 14px;")
        layout.addWidget(label)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sphere Diffraction")
        self.resize(1600, 900)
        self._build_menu()
        self._build_splitter()

    def _build_menu(self):
        menu = self.menuBar()
        file_menu = menu.addMenu("File")
        file_menu.addAction("New")
        file_menu.addAction("Open...")
        file_menu.addAction("Save")
        file_menu.addSeparator()
        file_menu.addAction("Exit").triggered.connect(self.close)
        menu.addMenu("View")

    def _build_splitter(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(6)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #555;
            }
            QSplitter::handle:hover {
                background-color: #aaa;
            }
        """)

        # Left
        left = QWidget()
        left.setStyleSheet("background-color: #12121a;")
        left_layout = QVBoxLayout(left)
        left_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        left_layout.setContentsMargins(8, 8, 8, 8)

        wave_group = QGroupBox("Wave parameters")
        wave_group.setStyleSheet("color: #ccc;")
        wave_form = QFormLayout(wave_group)

        wavelength_spin = QDoubleSpinBox()
        wavelength_spin.setRange(0.0001, 100.0)
        wavelength_spin.setDecimals(4)
        wavelength_spin.setSingleStep(0.01)
        wavelength_spin.setValue(0.5)
        wavelength_spin.setSuffix(" m")
        wave_form.addRow("Wavelength (λ):", wavelength_spin)

        left_layout.addWidget(wave_group)

        # Right
        right = QWidget()
        grid = QGridLayout(right)
        grid.setSpacing(6)
        grid.setContentsMargins(6, 6, 6, 6)
        for row in range(3):
            for col in range(2):
                grid.addWidget(PlaceholderPlot(row * 3 + col + 1), row, col)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([400, 800])

        self.setCentralWidget(splitter)