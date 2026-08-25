"""
Master Wrapper Launcher for Central PC Components (Review 1 Demo)

Single 1-Click Execution Script for PC:
1. Starts Central PC Master Dashboard & FL Aggregator Server (Port 5000)
2. Automatically opens visual dashboard in PC default browser (http://localhost:5000)
3. Launches Node 1 (Physical PC Node)
4. Launches Nodes 3, 4, 5 (3 Virtual Baseline Nodes)
"""

import os
import sys
import time
import subprocess
import threading
import webbrowser

def start_master_dashboard_server():
    pc_dir = os.path.dirname(os.path.abspath(__file__))
    dashboard_script = os.path.join(pc_dir, "master_dashboard_pc.py")
    subprocess.run([sys.executable, dashboard_script])

def launch_pc_nodes():
    pc_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Wait for server to bind port
    time.sleep(2.5)
    
    # 1. Open PC Master Dashboard UI in Browser
    print("\n[INFO] Opening Central PC Master Dashboard in browser (http://localhost:5000)...")
    try:
        webbrowser.open("http://localhost:5000")
    except Exception:
        pass

    node1_script = os.path.join(pc_dir, "node1_pc_physical.py")
    virtuals_script = os.path.join(pc_dir, "virtual_nodes_3_4_5.py")

    for round_num in range(1, 31):
        print(f"\n[INFO] >>> LAUNCHING PC NODES FOR ROUND {round_num} <<<")
        
        p1 = subprocess.Popen([sys.executable, node1_script])
        time.sleep(2)
        p2 = subprocess.Popen([sys.executable, virtuals_script])
        
        # Wait for the PC nodes to finish their epoch and upload weights
        p1.wait()
        p2.wait()
        
        print(f"\n[INFO] PC Nodes completed Round {round_num}. Waiting for Pi and Server Krum Aggregation...")
        time.sleep(5)

if __name__ == "__main__":
    print("=" * 70)
    print("  MASTER WRAPPER LAUNCHER -- REVIEW 1 PC DEMO")
    print("=" * 70)
    print("Starting Central Dashboard Server & FL Client Nodes...\n")
    
    # Thread 1: Central Dashboard Server
    server_thread = threading.Thread(target=start_master_dashboard_server, daemon=True)
    server_thread.start()
    
    # Thread 2: Launcher for Nodes & Browser
    launcher_thread = threading.Thread(target=launch_pc_nodes, daemon=True)
    launcher_thread.start()
    
    print("\n[SUCCESS] All PC components initialized successfully!")
    print("[INFO] Keep this terminal open during your presentation.")
    print("[INFO] Press Ctrl+C anytime to stop.\n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping PC Master Demo.")
