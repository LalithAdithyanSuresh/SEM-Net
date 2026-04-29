#!/bin/bash
echo "Installing Gunicorn if not present..."
pip install gunicorn

echo "Starting Validation Dashboard with Gunicorn..."
# 4 workers, timeout 120s for uploads
gunicorn -w 4 -b 0.0.0.0:5003 --timeout 120 app:app
