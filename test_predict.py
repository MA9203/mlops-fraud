import requests

url = "http://127.0.0.1:8000/predict"

sample = {
    "Time": 1000,
    "Amount": 50.0,
    "V1": -1.2,
    "V2": 0.5,
    "V3": 1.4,
    "V4": -0.7,
    "V5": 0.3,
    "V6": -0.2,
    "V7": 0.8,
    "V8": -0.5,
    "V9": 1.1,
    "V10": -0.9,
    "V11": 0.2,
    "V12": 0.1,
    "V13": -0.3,
    "V14": -1.1,
    "V15": 0.7,
    "V16": -0.4,
    "V17": 0.9,
    "V18": -1.0,
    "V19": 0.6,
    "V20": -0.8,
    "V21": 1.3,
    "V22": -0.6,
    "V23": 0.4,
    "V24": -0.2,
    "V25": 0.1,
    "V26": -0.9,
    "V27": 0.7,
    "V28": -1.4
}

print("Sending request...")
response = requests.post(url, json=sample)

print("STATUS =", response.status_code)
print("RAW RESPONSE =")
print(response.text)

# Try decode JSON only if status is OK
if response.status_code == 200:
    try:
        print("\nParsed JSON =", response.json())
    except:
        print("⚠️ Could not decode JSON.")
