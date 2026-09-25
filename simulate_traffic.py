import io
import time
import requests
from PIL import Image

API_URL = "http://localhost:5000"

def simulate_network_telemetry():
    """Simulates benign and anomalous network traffic ingestion."""
    print("\n--- Simulating Network Telemetry ---")
    
    # 1. Normal traffic
    normal_log = {
        "source_id": "sensor_cyber_1",
        "src_ip": "192.168.1.105",
        "dst_ip": "10.0.0.5",
        "src_port": 443,
        "dst_port": 50000,
        "protocol": "TCP",
        "packet_size": 1200,
        "packet_rate": 80.0,
        "flags": "ACK"
    }
    resp = requests.post(f"{API_URL}/api/v1/network/log", json=normal_log)
    print(f"Normal Log -> Status: {resp.status_code}, Enqueue Latency: {resp.json().get('enqueue_latency_ms')} ms")

    # 2. Anomalous traffic (port scan / unusual volume)
    anomaly_log = {
        "source_id": "sensor_cyber_1",
        "src_ip": "45.33.32.156",
        "dst_ip": "10.0.0.5",
        "src_port": 64222,
        "dst_port": 22,
        "protocol": "TCP",
        "packet_size": 48,
        "packet_rate": 4500.0,
        "flags": "SYN"
    }
    resp = requests.post(f"{API_URL}/api/v1/network/log", json=anomaly_log)
    print(f"Anomalous Log -> Status: {resp.status_code}, Enqueue Latency: {resp.json().get('enqueue_latency_ms')} ms")

def check_recorded_events():
    """Queries the events endpoint to verify flagged anomalies."""
    print("\n--- Querying Flagged Events ---")
    resp = requests.get(f"{API_URL}/api/v1/events?limit=5")
    if resp.status_code == 200:
        events = resp.json().get("events", [])
        print(f"Total events found: {len(events)}")
        for ev in events:
            print(f"  [ID {ev['event_id']}] Source: {ev['source_id']} | Risk: {ev['risk_score']} | Severity: {ev['severity']}")
    else:
        print(f"Failed to query events: {resp.status_code}")

if __name__ == "__main__":
    simulate_network_telemetry()
    check_recorded_events()
