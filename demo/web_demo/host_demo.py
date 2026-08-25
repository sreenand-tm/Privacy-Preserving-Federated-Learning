import http.server
import socketserver
import os
import threading
import subprocess
import re
import sys

PORT = 8000

class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

def start_server():
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        httpd.serve_forever()

if __name__ == "__main__":
    web_demo_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(web_demo_dir)
    
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    print(f"Local web server started at http://localhost:{PORT}")
    
    cloudflared_path = os.path.join(web_demo_dir, "cloudflared.exe")
    if os.path.exists(cloudflared_path):
        print("Launching Cloudflare Secure HTTPS Tunnel...")
        cmd = [cloudflared_path, "tunnel", "--url", f"http://localhost:{PORT}", "--protocol", "http2"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        tunnel_url = None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
            if match and not tunnel_url:
                tunnel_url = match.group(0)
                print("\n" + "="*65)
                print("  🎉 SECURE HTTPS EDGE NODE DEMO LINK (CLOUDFLARE TUNNEL)")
                print("="*65)
                print(f"\n  Open this link on your phone's browser:\n")
                print(f"  👉  {tunnel_url}\n")
                print("  ✅ Works on ANY Wi-Fi or 4G/5G mobile network!")
                print("  ✅ No Chrome Flags, zero security warnings!")
                print("  ✅ 100% Local Inference & Mel-Spectrogram on Phone Processor!")
                print("="*65 + "\n")
        
        proc.wait()
    else:
        print(f"Serving UI locally at http://0.0.0.0:{PORT}")
        print("Connect to your local IP address from your phone's browser.")
        server_thread.join()
