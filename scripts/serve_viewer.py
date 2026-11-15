"""Serve the viewer locally, with HTTP Range support so the browser can seek inside the video.

Binds to 127.0.0.1 only: the match video and tracking never leave this machine.

    python scripts/serve_viewer.py            # then open http://127.0.0.1:8765/mtah-possession/viewer/
"""
import http.server
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mtah import config  # noqa: E402

PORT = int(os.environ.get("VIEWER_PORT", "8765"))
CHUNK = 1 << 20


class RangeHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=config.WORKSPACE, **kwargs)

    def send_head(self):
        path = self.translate_path(self.path)
        match = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if not match or not os.path.isfile(path):
            return super().send_head()
        size = os.path.getsize(path)
        start = int(match.group(1)) if match.group(1) else max(0, size - int(match.group(2) or 0))
        end = int(match.group(2)) if match.group(1) and match.group(2) else size - 1
        end = min(end, size - 1)
        if start > end:
            self.send_error(416)
            return None
        handle = open(path, "rb")
        handle.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", "bytes {}-{}/{}".format(start, end, size))
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self._remaining = end - start + 1
        return handle

    def copyfile(self, source, outputfile):
        remaining = getattr(self, "_remaining", None)
        if remaining is None:
            return super().copyfile(source, outputfile)
        try:
            while remaining > 0:
                data = source.read(min(CHUNK, remaining))
                if not data:
                    break
                outputfile.write(data)
                remaining -= len(data)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self._remaining = None

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    server = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), RangeHandler)
    print("viewer at http://127.0.0.1:{}/mtah-possession/viewer/".format(PORT), flush=True)
    server.serve_forever()
