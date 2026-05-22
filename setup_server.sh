#!/usr/bin/env bash

# Redirect execution to the python script setup_server.py which manages the installation with a live progress bar
exec python3 "$(dirname "$0")/setup_server.py" "$@"
