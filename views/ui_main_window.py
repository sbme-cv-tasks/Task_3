from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QFile, QMetaObject, Qt
from PySide6.QtGui import QAction, QPixmap
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QFileDialog, QMainWindow, QMenu, QMenuBar, QStatusBar, QStyle


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        loader = QUiLoader()
        ui_path = Path(__file__).with_name("main_window.ui")

        ui_file = QFile(str(ui_path))
        if not ui_file.open(QFile.OpenModeFlag.ReadOnly):
            raise FileNotFoundError(f"Unable to open UI file: {ui_path}")

        self.centralwidget = loader.load(ui_file, MainWindow)
        ui_file.close()

        if self.centralwidget is None:
            raise RuntimeError(f"Failed to load UI file: {ui_path}")

        MainWindow.resize(1366, 768)
        MainWindow.setWindowTitle(self.centralwidget.windowTitle())
        MainWindow.setCentralWidget(self.centralwidget)

        self.actionOpen_Image = QAction(MainWindow)
        self.actionOpen_Image.setObjectName("actionOpen_Image")
        self.actionSave_Result = QAction(MainWindow)
        self.actionSave_Result.setObjectName("actionSave_Result")
        self.actionExit = QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")

        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName("menubar")
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menubar.addAction(self.menuFile.menuAction())
        MainWindow.setMenuBar(self.menubar)

        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menuFile.addAction(self.actionOpen_Image)
        self.menuFile.addAction(self.actionSave_Result)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionExit)

        self.topBar = self.centralwidget.findChild(type(self.centralwidget), "topBar")
        self.topTimeFrame = self.centralwidget.findChild(type(self.centralwidget), "topTimeFrame")
        self.lblMode = self.centralwidget.findChild(type(self.centralwidget), "lblMode")
        self.comboMode = self.centralwidget.findChild(type(self.centralwidget), "comboMode")
        self.btnUploadOriginal = self.centralwidget.findChild(type(self.centralwidget), "btnUploadOriginal")
        self.btnReset = self.centralwidget.findChild(type(self.centralwidget), "btnReset")
        self.lblTimeLabelSingle = self.centralwidget.findChild(type(self.centralwidget), "lblTimeLabelSingle")
        self.lblTimeValueSingle = self.centralwidget.findChild(type(self.centralwidget), "lblTimeValueSingle")

        self.displayStack = self.centralwidget.findChild(type(self.centralwidget), "displayStack")
        self.hDivider = self.centralwidget.findChild(type(self.centralwidget), "hDivider")
        self.parametersStack = self.centralwidget.findChild(type(self.centralwidget), "parametersStack")

        self.lblSingleOriginalTitle = self.centralwidget.findChild(type(self.centralwidget), "lblSingleOriginalTitle")
        self.lblSingleOutputTitle = self.centralwidget.findChild(type(self.centralwidget), "lblSingleOutputTitle")
        self.lblOriginal = self.centralwidget.findChild(type(self.centralwidget), "lblOriginal")
        self.lblProcessed = self.centralwidget.findChild(type(self.centralwidget), "lblProcessed")

        self.lblImg1Title = self.centralwidget.findChild(type(self.centralwidget), "lblImg1Title")
        self.lblImg2Title = self.centralwidget.findChild(type(self.centralwidget), "lblImg2Title")
        self.btnUploadMatchImage1 = self.centralwidget.findChild(type(self.centralwidget), "btnUploadMatchImage1")
        self.btnUploadMatchImage2 = self.centralwidget.findChild(type(self.centralwidget), "btnUploadMatchImage2")
        self.lblMatchImage1 = self.centralwidget.findChild(type(self.centralwidget), "lblMatchImage1")
        self.lblMatchImage2 = self.centralwidget.findChild(type(self.centralwidget), "lblMatchImage2")
        self.lblMatchResult = self.centralwidget.findChild(type(self.centralwidget), "lblMatchResult")
        self.matchTimeFrame = self.centralwidget.findChild(type(self.centralwidget), "matchTimeFrame")
        self.lblTimeLabelMatch = self.centralwidget.findChild(type(self.centralwidget), "lblTimeLabelMatch")
        self.lblTimeValueMatch = self.centralwidget.findChild(type(self.centralwidget), "lblTimeValueMatch")

        self.frameHarris = self.centralwidget.findChild(type(self.centralwidget), "frameHarris")
        self.frameSift = self.centralwidget.findChild(type(self.centralwidget), "frameSift")
        self.matchSideFrame = self.centralwidget.findChild(type(self.centralwidget), "matchSideFrame")

        self.lblHarris = self.centralwidget.findChild(type(self.centralwidget), "lblHarris")
        self.lblSift = self.centralwidget.findChild(type(self.centralwidget), "lblSift")
        self.lblMatchUsing = self.centralwidget.findChild(type(self.centralwidget), "lblMatchUsing")

        self.harrisThreshold = self.centralwidget.findChild(type(self.centralwidget), "harrisThreshold")
        self.harrisWindow = self.centralwidget.findChild(type(self.centralwidget), "harrisWindow")
        self.lam_min_check = self.centralwidget.findChild(type(self.centralwidget), "lam_min_check")
        self.siftOctaves = self.centralwidget.findChild(type(self.centralwidget), "siftOctaves")
        self.matchThreshold = self.centralwidget.findChild(type(self.centralwidget), "matchThreshold")
        self.matchCount = self.centralwidget.findChild(type(self.centralwidget), "matchCount")
        self.radioNCC = self.centralwidget.findChild(type(self.centralwidget), "radioNCC")
        self.radioSSD = self.centralwidget.findChild(type(self.centralwidget), "radioSSD")
        self.btnApplyHarris = self.centralwidget.findChild(type(self.centralwidget), "btnApplyHarris")
        self.btnApplySift = self.centralwidget.findChild(type(self.centralwidget), "btnApplySift")
        self.btnMatchNCC = self.centralwidget.findChild(type(self.centralwidget), "btnMatchNCC")
        self.btnMatchSSD = self.centralwidget.findChild(type(self.centralwidget), "btnMatchSSD")

        self._set_match_upload_icons(MainWindow)
        self._wire_signals(MainWindow)
        self._sync_mode_widgets(0)
        QMetaObject.connectSlotsByName(MainWindow)

    def _wire_signals(self, MainWindow):
        # self.actionOpen_Image.triggered.connect(self._upload_original_image)
        # self.btnUploadOriginal.clicked.connect(self.actionOpen_Image.trigger)
        self.btnReset.clicked.connect(self._reset_view_labels)
        self.comboMode.currentIndexChanged.connect(self._sync_mode_widgets)
        if self.btnUploadMatchImage1 is not None:
            self.btnUploadMatchImage1.clicked.connect(lambda: self._upload_match_image(self.lblMatchImage1))
        if self.btnUploadMatchImage2 is not None:
            self.btnUploadMatchImage2.clicked.connect(lambda: self._upload_match_image(self.lblMatchImage2))

    def _set_match_upload_icons(self, MainWindow):
        open_icon = MainWindow.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton)
        if self.btnUploadMatchImage1 is not None:
            self.btnUploadMatchImage1.setIcon(open_icon)
        if self.btnUploadMatchImage2 is not None:
            self.btnUploadMatchImage2.setIcon(open_icon)

    def _upload_original_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.centralwidget,
            "Open Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp)",
        )
        if not file_path:
            return

        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            self.statusbar.showMessage("Unable to load selected image.", 2000)
            return

        self.lblOriginal.setPixmap(
            pixmap.scaled(self.lblOriginal.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        self.lblOriginal.setText("")
        self.statusbar.showMessage(f"Loaded: {file_path}", 2000)

    def _upload_match_image(self, target_label):
        file_path, _ = QFileDialog.getOpenFileName(
            self.centralwidget,
            "Open Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp)",
        )
        if not file_path:
            return

        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            self.statusbar.showMessage("Unable to load selected image.", 2000)
            return

        target_label.setPixmap(
            pixmap.scaled(target_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        target_label.setProperty("imagePath", file_path)
        target_label.setText("")
        self.statusbar.showMessage(f"Loaded: {file_path}", 2000)

    def _reset_view_labels(self):
        self.lblOriginal.setText("No Image")
        self.lblProcessed.setText("No Image")
        self.lblMatchImage1.setText("No Image")
        self.lblMatchImage2.setText("No Image")
        self.lblMatchResult.setText("Matching Result")
        self.lblTimeValueSingle.setText("0.000 S")
        self.lblTimeValueMatch.setText("0.000 S")
        self.statusbar.showMessage("View reset.", 2000)

    def _sync_mode_widgets(self, index):
        self.parametersStack.setCurrentIndex(index)
        self.displayStack.setCurrentIndex(1 if index == 2 else 0)
        if self.matchTimeFrame is not None:
            self.matchTimeFrame.setVisible(index == 2)
        if index == 2:
            self.hDivider.hide()
            self.parametersStack.hide()
        else:
            self.hDivider.show()
            self.parametersStack.show()
