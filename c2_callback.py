import os
import sys
import glob
import logging
import requests
import pytorch_lightning as ptl

LOGGER = logging.getLogger(__name__)

class C2Callback(ptl.Callback):
    def __init__(self, server_url, visualizer_outdir, check_interval=50):
        super().__init__()
        self.server_url = server_url.rstrip('/')
        self.visualizer_outdir = visualizer_outdir
        self.check_interval = check_interval
        self.uploaded_images = set()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        pass

    def on_train_epoch_end(self, trainer, pl_module):
        pass

    def on_validation_epoch_end(self, trainer, pl_module):
        pass

    def _poll_command(self, trainer):
        pass
