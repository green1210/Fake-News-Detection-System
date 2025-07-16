import requests
import json

# The running API URL
URL = "http://127.0.0.1:8000/predict"

print("--- Testing English ---")
payload_en = {
    "text": "New study shows correlation between coffee consumption and productivity.",
    "language": "en"
}
response_en = requests.post(URL, json=payload_en)
print(f"Status Code: {response_en.status_code}")
print(f"Response: {response_en.json()}")


print("\n--- Testing Telugu ---")
payload_te = {
    "text": "ప్రభుత్వం రైతులకు కొత్త పథకాలను ప్రకటించింది.",
    "language": "te"
}
response_te = requests.post(URL, json=payload_te)
print(f"Status Code: {response_te.status_code}")
print(f"Response: {response_te.json()}")