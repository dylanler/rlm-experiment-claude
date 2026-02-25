#!/usr/bin/env python3
"""Static site server for the Latent Pager Memory experiment report."""
import http.server
import socketserver
import os
import sys

PORT = 8766

os.chdir(os.path.dirname(os.path.abspath(__file__)))


class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {format % args}")

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    with socketserver.TCPServer(("0.0.0.0", port), Handler) as httpd:
        print(f"Serving experiment report at http://0.0.0.0:{port}")
        print(f"Open in browser: http://10.1.7.101:{port}")
        httpd.serve_forever()
