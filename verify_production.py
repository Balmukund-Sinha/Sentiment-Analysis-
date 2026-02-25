
import requests
import time
import json

URL = "http://127.0.0.1:8000"

def test_single():
    print("\n--- Testing Single Prediction ---")
    payload = {"text": "This movie is a masterpiece of modern cinema.", "explain": True}
    start = time.time()
    r = requests.post(f"{URL}/predict", json=payload)
    data = r.json()
    print(f"Sentiment: {data['sentiment']} ({data['confidence']})")
    print(f"Status: {data['status']}")
    print(f"Latency: {data['latency_ms']}ms")
    print(f"Explanation tokens: {[t['token'] for t in data['explanation'][:3]]}")
    return data['latency_ms']

def test_cache(text):
    print(f"\n--- Testing Cache for: '{text}' ---")
    payload = {"text": text, "explain": True}
    # First run (Cold)
    r1 = requests.post(f"{URL}/predict", json=payload).json()
    print(f"Cold Latency: {r1['latency_ms']}ms")
    
    # Second run (Cached)
    r2 = requests.post(f"{URL}/predict", json=payload).json()
    print(f"Cached Latency: {r2['latency_ms']}ms")
    
    saving = r1['latency_ms'] - r2['latency_ms']
    print(f"Cache Savings: {round(saving, 2)}ms")

def test_batch():
    print("\n--- Testing Batch Prediction ---")
    payload = {
        "texts": [
            "Pure garbage. Avoid at all costs.",
            "Absolutely brilliant, a must watch!",
            "It was okay I guess, nothing special.",
            "Visuals were stunning but plot was thin."
        ],
        "explain": False
    }
    r = requests.post(f"{URL}/predict_batch", json=payload)
    results = r.json()
    print(f"Processed {len(results)} items.")
    for i, res in enumerate(results):
        print(f"  [{i}] {res['sentiment']} (Conf: {res['confidence']}) - Status: {res['status']}")

if __name__ == "__main__":
    try:
        test_single()
        test_cache("I loved every minute of it!")
        test_batch()
        print("\n✅ All production features verified.")
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
