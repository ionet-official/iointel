import requests

# Replace with your server's host and port if different
url = "http://0.0.0.0:8000/schedule"

# Schedule a reminder task to "Remind me to go jogging" in 5 minutes (300 seconds)
payload = {
    "command": "Remind me to go jogging",
    "delay": 30
}

response = requests.post(url, json=payload)
print(response.json())