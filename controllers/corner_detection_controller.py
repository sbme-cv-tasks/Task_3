from core.corner_detector import corner_detector


class CornerDetectionController:
    def __init__(self,ui,model,display_callback):

        self.ui = ui
        self.model = model
        self.display_callback = display_callback
        self._connect_signals()


    def _connect_signals(self):
        # Connect the Apply button from the UI to the method in this class
        self.ui.btnApplyHarris.clicked.connect(self.run_corner_detection)

    def run_corner_detection(self):

        # Error handling
        if self.model.original_image is None:
            self.ui.statusbar.showMessage("Please load an image first!", 3000)
            return

        # Get parameters from UI
        harris_threshold = self.ui.harrisThreshold.value()
        # harris_sigma= self.ui.harrisSigma.value()
        harris_window = self.ui.harrisWindow.value()
        lambda_minus_flag = self.ui.lam_min_check.isChecked()

        # scaled_threshold = harris_threshold * 10000
        # print(scaled_threshold)
        # Apply Corner Detection
        _,corners,elapsed_time = corner_detector(self.model.original_image,
                                    # harris_sigma,
                                    window_size = harris_window,
                                    threshold = harris_threshold,
                                    lambda_minus_flag = lambda_minus_flag
                                    )


        self.model.processed_image = corners
        self.display_callback(self.model.processed_image)
        self.ui.lblTimeValueSingle.setText(f"{elapsed_time:.3f} S")
        self.ui.statusbar.showMessage(f"Applied Corner Detection", 3000)


