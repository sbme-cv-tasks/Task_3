from core.SIFT import run_sift_pipeline


class SIFTController:
    def __init__(self, ui, model, display_callback):
        self.ui = ui
        self.model = model
        self.display_callback = display_callback
        self._connect_signals()

    def _connect_signals(self):
        self.ui.btnApplySift.clicked.connect(self.run_sift)

    def run_sift(self):
        if self.model.original_image is None:
            self.ui.statusbar.showMessage("Please load an image first!", 3000)
            return

        octaves = int(self.ui.siftOctaves.value())

        keypoints, output_img, elapsed_time = run_sift_pipeline(
            self.model.original_image,
            num_octaves=octaves,
        )

        self.model.processed_image = output_img
        self.display_callback(self.model.processed_image)
        self.ui.lblTimeValueSingle.setText(f"{elapsed_time:.3f} S")
        self.ui.statusbar.showMessage(f"SIFT completed: {len(keypoints)} keypoints", 3000)
