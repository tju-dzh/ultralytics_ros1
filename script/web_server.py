#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple HTTP server for Ground Station web interface

This script serves the ground station HTML interface and provides
a simple web server for remote monitoring.

Usage:
    python3 web_server.py --port 8000 --address 0.0.0.0
"""

import http.server
import socketserver
import os
import sys
import argparse
from pathlib import Path


class GroundStationHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP request handler for ground station"""

    def __init__(self, *args, directory=None, **kwargs):
        if directory is None:
            # Get the web_interface directory
            script_dir = Path(__file__).parent
            directory = script_dir.parent / "web_interface"

        super().__init__(*args, directory=str(directory), **kwargs)

    def end_headers(self):
        # Enable CORS for cross-origin requests
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def log_message(self, format, *args):
        # Custom log format
        sys.stdout.write("[Ground Station] %s - %s\n" % (
            self.address_string(),
            format % args
        ))


def main():
    parser = argparse.ArgumentParser(
        description="Ground Station Web Server for Multi-Camera Tracking System"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to serve on (default: 8000)"
    )
    parser.add_argument(
        "--address",
        type=str,
        default="0.0.0.0",
        help="Address to bind to (default: 0.0.0.0 for all interfaces)"
    )
    parser.add_argument(
        "--directory",
        type=str,
        default=None,
        help="Directory to serve (default: ../web_interface)"
    )

    args = parser.parse_args()

    # Set directory
    if args.directory:
        web_dir = Path(args.directory)
    else:
        script_dir = Path(__file__).parent
        web_dir = script_dir.parent / "web_interface"

    if not web_dir.exists():
        print(f"Error: Web interface directory not found: {web_dir}")
        sys.exit(1)

    # Change to web directory
    os.chdir(web_dir)

    # Create server
    Handler = lambda *args, **kwargs: GroundStationHTTPRequestHandler(
        *args, directory=str(web_dir), **kwargs
    )

    with socketserver.TCPServer((args.address, args.port), Handler) as httpd:
        print("=" * 80)
        print("Ground Station Web Server")
        print("=" * 80)
        print(f"Server address: {args.address}")
        print(f"Server port: {args.port}")
        print(f"Serving directory: {web_dir}")
        print()
        print(f"Access the ground station at:")
        print(f"  http://localhost:{args.port}/ground_station.html")
        print(f"  http://<AGX_ORIN_IP>:{args.port}/ground_station.html")
        print()
        print("Press Ctrl+C to stop the server")
        print("=" * 80)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            httpd.shutdown()


if __name__ == "__main__":
    main()
