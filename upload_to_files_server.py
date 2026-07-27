#!/usr/bin/env python3
import os
import sys
import argparse
import requests
import time

def upload_file_chunked(file_path, server_url, session_id, chunk_size=10 * 1024 * 1024):
    print("[UPLOAD] Network file uploads are disabled.")
    return False

def main():
    parser = argparse.ArgumentParser(description="Upload any file to the files server.")
    parser.add_argument("file_path", help="Path to the file to upload")
    parser.add_argument("--server", default="https://files.lalithadithyan.dev", help="Files server base URL")
    parser.add_argument("--session", default="DAVA", help="Session ID")
    parser.add_argument("--chunk-size", type=int, default=10, help="Chunk size in MB (default: 10)")
    
    args = parser.parse_args()
    
    chunk_bytes = args.chunk_size * 1024 * 1024
    upload_file_chunked(args.file_path, args.server, args.session, chunk_bytes)

if __name__ == "__main__":
    main()
