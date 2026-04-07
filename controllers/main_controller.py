import cv2
from PySide6.QtWidgets import QFileDialog
from PySide6.QtCore import Qt
from utils.converters import cv_to_pixmap
from controllers.corner_detection_controller import CornerDetectionController
from controllers.sift_controller import SIFTController


class MainController:
    def __init__(self, ui, model, window):
        self.ui = ui
        self.model = model
        self.window = window

        self.CornerDetectionController = CornerDetectionController(self.ui, self.model,display_callback=self.display_processed_image)
        self.SIFTController = SIFTController(self.ui, self.model, display_callback=self.display_processed_image)

        self._connect_signals()

    def _connect_signals(self):
        # File menu
        self.ui.actionOpen_Image.triggered.connect(self.load_image)
        # self.ui.actionSave_Result.triggered.connect(self.save_result)
        self.ui.actionExit.triggered.connect(self.window.close)

        # Top-bar buttons
        self.ui.btnUploadOriginal.clicked.connect(self.load_image)
        # self.ui.btnReset.clicked.connect(self.reset_view)


    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.window, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if not file_path:
            return

        self.model.original_image = cv2.imread(file_path)
        self.model.processed_image = None

        self.display_original_image(self.model.original_image)

        self.ui.lblProcessed.clear()
        self.ui.lblProcessed.setText("Ready for processing.")
        self.ui.statusbar.showMessage(f"Loaded: {file_path}", 3000)

    def display_original_image(self, cv_img):
        if cv_img is None:
            return

        pixmap = cv_to_pixmap(cv_img)
        if pixmap:
            # Scale the pixmap to fit the label while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.ui.lblOriginal.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.ui.lblOriginal.setPixmap(scaled_pixmap)

    def display_processed_image(self, cv_img):

        if cv_img is None:
            return

        pixmap = cv_to_pixmap(cv_img)
        if pixmap:
            # Scale the pixmap to fit the label while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.ui.lblProcessed.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.ui.lblProcessed.setPixmap(scaled_pixmap)

