#!/usr/bin/env python3
"""
Dashboard server for Latent Pager experiment.
Serves the HTML dashboard and provides API endpoints for log/result data.
"""

import http.server
import os
import json

PORT = 8765
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Strip query params
        path = self.path.split("?")[0]

        # Serve dashboard
        if path == "/" or path == "/index.html":
            self.serve_file(os.path.join(BASE_DIR, "dashboard", "index.html"), "text/html")
            return

        # Serve log files
        if path.startswith("/logs/"):
            log_path = os.path.join(BASE_DIR, "logs", path[6:])
            if os.path.exists(log_path):
                self.serve_file(log_path, "text/plain")
            else:
                self.send_error(404)
            return

        # Serve result data files
        if path.startswith("/data/"):
            data_path = os.path.join(BASE_DIR, "results", path[6:])
            if os.path.exists(data_path):
                content_type = "application/json" if path.endswith(".json") else "text/plain"
                self.serve_file(data_path, content_type)
            else:
                self.send_error(404)
            return

        # Serve status endpoint
        if path == "/api/status":
            self.serve_status()
            return

        self.send_error(404)

    def serve_file(self, filepath, content_type):
        try:
            with open(filepath, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", len(content))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, str(e))

    def serve_status(self):
        """Quick status check of running processes."""
        import subprocess
        result = subprocess.run(
            ["ps", "aux"], capture_output=True, text=True
        )
        running = []
        for line in result.stdout.split("\n"):
            if "scripts/0" in line and "python" in line and "grep" not in line:
                parts = line.split()
                running.append({
                    "pid": parts[1],
                    "cpu": parts[2],
                    "mem": parts[3],
                    "cmd": " ".join(parts[10:])
                })

        status = {
            "running_processes": running,
            "timestamp": __import__("datetime").datetime.now().isoformat(),
        }
        content = json.dumps(status).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(content))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format, *args):
        pass  # Suppress access logs


if __name__ == "__main__":
    os.chdir(BASE_DIR)
    server = http.server.HTTPServer(("0.0.0.0", PORT), DashboardHandler)
    print(f"Dashboard running at http://0.0.0.0:{PORT}")
    print(f"  Local: http://localhost:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down dashboard")
        server.shutdown()
