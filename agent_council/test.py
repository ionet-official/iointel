import httpx


def schedule():
    payload = {
        "task": "Remind me to give my mom a hug in 30 seconds"
    }

    print("Sending request...")
    response = httpx.post("http://0.0.0.0:8000/schedule", json=payload)
    print("Response received:")
    return response.json()



def council():
    payload = {
        "task": "write me some code to generate a synthetic data set and run logistic regression on it",
    }

    print("Sending request...")
    response = httpx.post("http://0.0.0.0:8000/council", json=payload)
    print("Response received:")
    return response.json()


if __name__ == "__main__":
    #print(schedule())
    print(council())