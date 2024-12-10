import httpx

payload = {
    "command": "Remind me to give my mom a hug",
    "delay": 30
}

print("Sending reminder request...")
response = httpx.post("http://0.0.0.0:8000/schedule", json=payload)
print("Response received:")
print(response.json())