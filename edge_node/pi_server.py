"""
pi_server.py — Raspberry Pi Sound Classification Server + UART to TM4C123
===========================================================================
Continuously classifies ambient sound using pw-record (PipeWire Bluetooth mic),
sends the detected class over UART to TM4C123GH6PM for I2C LCD display,
prints live results in the terminal, and serves a web dashboard for mobile.

Usage:
  Live Mic Mode:    python3 pi_server.py
  Demo WAV Mode:    python3 pi_server.py --demo path/to/audio.wav
  Custom Port:      python3 pi_server.py --port 9000
  Custom Model:     python3 pi_server.py --model path/to/model.onnx
  No UART (web only): python3 pi_server.py --no-uart
  Custom UART port: python3 pi_server.py --uart-port /dev/ttyACM0
"""

import os
import sys
import json
import time
import socket
import argparse
import threading
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler

# ─── Import the classifier ──────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from classify import EdgeClassifier, record_from_mic, load_wav_file, DURATION

# ─── Global State ────────────────────────────────────────────────────────────
current_result = {
    "class":       "background_noise",
    "label":       "Background Noise",
    "confidence":  0.0,
    "icon":        "🤫",
    "timestamp":   datetime.now().isoformat(),
    "mode":        "initializing",
    "history":     []
}
result_lock = threading.Lock()
classifier = None
uart_serial = None       # Global serial port handle (None if UART disabled)
MAX_HISTORY = 20


# ─── UART Helper ─────────────────────────────────────────────────────────────

def uart_init(port='/dev/serial0', baudrate=9600):
    """Open the UART serial port to the TM4C123GH6PM. Returns None on failure."""
    try:
        import serial
        ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
            timeout=1.0
        )
        print(f"  [UART] Opened {port} at {baudrate} baud  ✓")
        return ser
    except ImportError:
        print("  [UART] pyserial not installed! Run: pip3 install pyserial")
        print("  [UART] Continuing without UART...")
        return None
    except Exception as e:
        print(f"  [UART] Could not open {port}: {e}")
        print("  [UART] Continuing without UART...")
        return None


def uart_send(label):
    """Send the noise label string + newline to TM4C123 over UART."""
    global uart_serial
    if uart_serial is None:
        return
    try:
        # Truncate to 16 chars to fit on one row of 16x2 LCD
        clean = label.strip()[:16]
        uart_serial.write(f"{clean}\n".encode('utf-8'))
        uart_serial.flush()
    except Exception as e:
        print(f"  [UART] Write error: {e}")


# ─── Classification Loops ────────────────────────────────────────────────────

def classify_loop_mic():
    """Continuous live microphone classification loop (pw-record / PipeWire)."""
    global current_result
    print(f"\n  [Loop] Starting LIVE microphone classification...")
    print(f"  [Loop] Recording {DURATION}s windows continuously\n")

    while True:
        try:
            # Tell TM4C we are recording -> Green LED
            uart_send("RECORDING")

            audio = record_from_mic(DURATION)
            cls, conf, label, icon = classifier.predict(audio)

            entry = {
                "class": cls,
                "label": label,
                "confidence": round(conf, 4),
                "icon":  icon,
                "timestamp": datetime.now().isoformat()
            }

            with result_lock:
                current_result["class"]      = cls
                current_result["label"]      = label
                current_result["confidence"] = round(conf, 4)
                current_result["icon"]       = icon
                current_result["timestamp"]  = entry["timestamp"]
                current_result["mode"]       = "live_mic"
                current_result["history"].insert(0, entry)
                current_result["history"] = current_result["history"][:MAX_HISTORY]

            # Print to terminal
            print(f"  {icon} {label:20s}  {conf*100:5.1f}%  [{entry['timestamp'][11:19]}]")

            # Send to TM4C123 via UART
            uart_send(label)

        except Exception as e:
            print(f"  [Error] Mic recording failed: {e}")
            time.sleep(2)


def classify_loop_demo(wav_path):
    """Demo mode: classify the same WAV file in a loop."""
    global current_result
    print(f"\n  [Loop] Starting DEMO mode with: {os.path.basename(wav_path)}")
    print(f"  [Loop] Re-classifying every {DURATION + 1}s\n")

    audio = load_wav_file(wav_path)

    while True:
        try:
            cls, conf, label, icon = classifier.predict(audio)

            entry = {
                "class": cls,
                "label": label,
                "confidence": round(conf, 4),
                "icon":  icon,
                "timestamp": datetime.now().isoformat()
            }

            with result_lock:
                current_result["class"]      = cls
                current_result["label"]      = label
                current_result["confidence"] = round(conf, 4)
                current_result["icon"]       = icon
                current_result["timestamp"]  = entry["timestamp"]
                current_result["mode"]       = "demo_wav"
                current_result["history"].insert(0, entry)
                current_result["history"] = current_result["history"][:MAX_HISTORY]

            # Print to terminal
            print(f"  {icon} {label:20s}  {conf*100:5.1f}%  [{entry['timestamp'][11:19]}]")

            # Send to TM4C123 via UART
            uart_send(label)

            time.sleep(DURATION + 1)

        except Exception as e:
            print(f"  [Error] Demo classification failed: {e}")
            time.sleep(2)


# ─── HTTP Handler ────────────────────────────────────────────────────────────

class PiHandler(SimpleHTTPRequestHandler):
    """Serves the web dashboard and JSON classification API."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=SCRIPT_DIR, **kwargs)

    def do_GET(self):
        # ── JSON API ──
        if self.path == "/api/classification":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            with result_lock:
                payload = json.dumps(current_result)
            self.wfile.write(payload.encode("utf-8"))
            return

        # ── Dashboard + static files ──
        if self.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, fmt, *args):
        # Suppress noisy request logs (keep classify loop output clean)
        pass


# ─── Network Utils ───────────────────────────────────────────────────────────

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    global classifier, uart_serial

    parser = argparse.ArgumentParser(description="Pi Sound Classification Server + UART to TM4C123")
    parser.add_argument("--port", type=int, default=8000, help="HTTP server port (default: 8000)")
    parser.add_argument("--model", type=str, default=None, help="Path to ONNX model file")
    parser.add_argument("--demo", type=str, default=None, help="Path to WAV file for demo mode (no mic needed)")
    parser.add_argument("--uart-port", type=str, default="/dev/serial0", help="UART serial port (default: /dev/serial0)")
    parser.add_argument("--baud", type=int, default=9600, help="UART baud rate (default: 9600)")
    parser.add_argument("--no-uart", action="store_true", help="Disable UART transmission to TM4C123")
    args = parser.parse_args()

    print("=" * 60)
    print("   Raspberry Pi Sound Classification Server")
    print("   + UART → TM4C123GH6PM → I2C LCD Display")
    print("=" * 60)

    # ── Load classifier ──
    print("\n  Loading ONNX model...")
    classifier = EdgeClassifier(model_path=args.model)

    # ── Initialize UART ──
    if not args.no_uart:
        uart_serial = uart_init(port=args.uart_port, baudrate=args.baud)
    else:
        print("  [UART] Disabled (--no-uart flag)")
        uart_serial = None

    # ── Validate demo file ──
    if args.demo:
        if not os.path.exists(args.demo):
            print(f"\n  [Error] Demo WAV file not found: {args.demo}")
            sys.exit(1)
        print(f"  [Mode] DEMO — Using: {args.demo}")
    else:
        print(f"  [Mode] LIVE MIC — Recording via PipeWire pw-record")

    # ── Start classification thread ──
    if args.demo:
        t = threading.Thread(target=classify_loop_demo, args=(args.demo,), daemon=True)
    else:
        t = threading.Thread(target=classify_loop_mic, daemon=True)
    t.start()

    # ── Start HTTP server for mobile dashboard ──
    local_ip = get_local_ip()
    server = HTTPServer(("", args.port), PiHandler)

    print(f"\n  {'─' * 54}")
    print(f"  Server running on port {args.port}")
    print(f"  {'─' * 54}")
    print(f"  Dashboard (mobile): http://{local_ip}:{args.port}")
    print(f"  API endpoint:       http://{local_ip}:{args.port}/api/classification")
    if uart_serial:
        print(f"  UART → TM4C123:     {args.uart_port} @ {args.baud} baud")
    print(f"  {'─' * 54}")
    print(f"  Terminal output + web dashboard active")
    print(f"  Press Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down...")
        uart_send("STOPPED")
        time.sleep(0.1)
        server.shutdown()
        if uart_serial:
            uart_serial.close()
            print("  [UART] Port closed.")


if __name__ == "__main__":
    main()
