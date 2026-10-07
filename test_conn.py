# test_conn.py
import sys
import socket

def run_server(port=29500):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('0.0.0.0', port))
    s.listen(1)
    print(f"DEBUG: Server is listening on 0.0.0.0:{port}...")
    print("Waiting for client connection...")
    conn, addr = s.accept()
    print(f"SUCCESS: Connected by {addr}")
    conn.close()

def run_client(host, port=29500):
    print(f"DEBUG: Attempting to connect to {host}:{port}...")
    try:
        s = socket.create_connection((host, port), timeout=5)
        print("SUCCESS: Connection established!")
        s.close()
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_conn.py server [port]")
        print("       python test_conn.py client <host> [port]")
    elif sys.argv[1] == "server":
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 29500
        run_server(port)
    elif sys.argv[1] == "client":
        host = sys.argv[2]
        port = int(sys.argv[3]) if len(sys.argv) > 3 else 29500
        run_client(host, port)
