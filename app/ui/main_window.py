# main_window.py
from PyQt6.QtWidgets import (
    QMainWindow, QSplitter, QWidget,
    QGridLayout, QFrame, QLabel,
)
from PyQt6.QtCore import Qt

from .parameter_panel import ParameterPanel
from .ui_config import UIConfig, LIGHT_THEME
from .settings_dialog import SettingsDialog


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
    def __init__(self, controller=None):
        super().__init__()
        self._controller = controller

        # Build the UI config once, DPI-aware.
        # Swap LIGHT_THEME → DARK_THEME (or any custom palette) here
        # if you want a different startup theme.
        self._ui_cfg = UIConfig.from_screen(theme=LIGHT_THEME)

        self.setWindowTitle("Sphere Diffraction")
        self.resize(1600, 900)
        self._build_menu()
        self._build_splitter()

    # ------------------------------------------------------------------ #
    # Menu                                                                 #
    # ------------------------------------------------------------------ #

    def _build_menu(self):
        menu = self.menuBar()

        file_menu = menu.addMenu("File")
        file_menu.addAction("Load")
        file_menu.addAction("Save")
        file_menu.addSeparator()
        file_menu.addAction("Exit").triggered.connect(self.close)

        settings_menu = menu.addMenu("Settings")
        settings_menu.addAction("Preferences…").triggered.connect(
            self._open_settings
        )

        menu.addMenu("About")

    # ------------------------------------------------------------------ #
    # Central widget                                                       #
    # ------------------------------------------------------------------ #

    def _build_splitter(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(6)
        splitter.setStyleSheet("""
            QSplitter::handle { background-color: #555; }
            QSplitter::handle:hover { background-color: #aaa; }
        """)

        self.param_panel = ParameterPanel(
            controller=self._controller,
            config=self._ui_cfg,
        )

        if self._controller is not None:
            self.param_panel.wavelength_changed.connect(
                lambda v: self._controller.set_wave_length(v)
            )
            self.param_panel.parameters_changed.connect(
                self._on_parameters_changed
            )

        right = QWidget()
        grid = QGridLayout(right)
        grid.setSpacing(6)
        grid.setContentsMargins(6, 6, 6, 6)
        for row in range(3):
            for col in range(2):
                grid.addWidget(PlaceholderPlot(row * 2 + col + 1), row, col)

        splitter.addWidget(self.param_panel)
        splitter.addWidget(right)
        splitter.setSizes([400, 800])

        self.setCentralWidget(splitter)

    # ------------------------------------------------------------------ #
    # Settings                                                             #
    # ------------------------------------------------------------------ #

    def _open_settings(self):
        dlg = SettingsDialog(current_config=self._ui_cfg, parent=self)
        if dlg.exec():
            self._apply_config(dlg.result_config())

    def _apply_config(self, config: UIConfig):
        """Store new config and push it to every themed widget."""
        self._ui_cfg = config
        self.param_panel.apply_config(config)
        # If other panels / widgets grow to accept UIConfig, call them here.

    # ------------------------------------------------------------------ #
    # Parameter changes                                                    #
    # ------------------------------------------------------------------ #

    def _on_parameters_changed(self):
        """Any parameter changed — trigger recompute if auto-refresh is on."""
        if self._controller and self.param_panel.get_auto_refresh():
            wavelength = self.param_panel.get_wavelength()
            layers     = self.param_panel.get_layers()
            fidelity   = self.param_panel.get_fidelity()
            # self._controller.compute(wavelength, layers, fidelity)
            pass