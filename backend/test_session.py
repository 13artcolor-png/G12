
import requests
import json
import time

API_URL = "http://localhost:8012"

def test_full_session_flow():
    print(f"Testing ULTRA COMPLETE session flow on {API_URL}...")
    
    try:
        # 1. Start session
        print("\n1. POST /api/session/start")
        r = requests.post(f"{API_URL}/api/session/start")
        print(f"Status: {r.status_code}, Response: {r.json()}")
        
        # 2. Simulate some trades (via direct file write or if there's an API)
        # For now, we trust the logger.log_trade works during real trading.
        # But we can check if /api/session shows info
        print("\n2. GET /api/session")
        r = requests.get(f"{API_URL}/api/session")
        print(f"Data: {json.dumps(r.json(), indent=2)}")
        
        # 3. End session (This triggers the ultra report)
        print("\n3. POST /api/session/end")
        r = requests.post(f"{API_URL}/api/session/end")
        print(f"Status: {r.status_code}")
        data = r.json()
        if data.get('success'):
            print("SUCCESS: Session ended.")
            print(f"Log file: {data.get('log_file')}")
            print(f"Txt file: {data.get('txt_file')}")
            print(f"Summary: {json.dumps(data.get('summary'), indent=2)}")
        else:
            print(f"FAILED: {data.get('message')}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_full_session_flow()
