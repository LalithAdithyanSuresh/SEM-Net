#!/usr/bin/env bash

# Live interactive command loop
# This script runs indefinitely and executes any command you type.
# It does not start any training or other processes.
# To exit, press Ctrl+C.

while true; do
  read -p "[commandline]$ " cmd
  # If the user enters an empty line, just continue
  if [[ -z "$cmd" ]]; then
    continue
  fi
  # Execute the command and capture exit status
  eval $cmd
  echo "[commandline] exit status: $?"
done
