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
        if batch_idx % self.check_interval == 0:
            self._poll_command(trainer)

    def on_train_epoch_end(self, trainer, pl_module):
        self._poll_command(trainer)

    def on_validation_epoch_end(self, trainer, pl_module):
        # 1. Upload Metrics
        metrics = {k: float(v.cpu().item()) if hasattr(v, 'item') else float(v) 
                   for k, v in trainer.logged_metrics.items()}
        metrics['epoch'] = trainer.current_epoch
        
        try:
            requests.post(f"{self.server_url}/api/metrics", json=metrics, timeout=5)
        except Exception as e:
            LOGGER.warning(f"Failed to push metrics to C2 Server: {e}")

        # 2. Upload New Images
        try:
            if not os.path.exists(self.visualizer_outdir):
                return
            
            # Find all images
            images = glob.glob(os.path.join(self.visualizer_outdir, '**', '*.png'), recursive=True)
            images += glob.glob(os.path.join(self.visualizer_outdir, '**', '*.jpg'), recursive=True)
            
            for img_path in images:
                if img_path not in self.uploaded_images:
                    with open(img_path, 'rb') as f:
                        files = {'file': (os.path.basename(img_path), f, 'image/png')}
                        res = requests.post(f"{self.server_url}/api/upload_image", files=files, timeout=10)
                        if res.status_code == 200:
                            self.uploaded_images.add(img_path)
        except Exception as e:
            LOGGER.warning(f"Failed to upload images to C2 Server: {e}")

    def _poll_command(self, trainer):
        try:
            res = requests.get(f"{self.server_url}/api/command", timeout=2)
            if res.status_code == 200:
                data = res.json()
                cmd = data.get('command', 'run')
                if cmd == 'stop':
                    LOGGER.info("C2 Server requested stop. Halting training gracefully.")
                    trainer.should_stop = True
                elif cmd == 'restart_pull':
                    LOGGER.info("C2 Server requested git pull & restart! Exiting with code 42.")
                    # Let the bash wrapper handle the restart
                    sys.exit(42)
        except Exception:
            pass # ignore timeouts
