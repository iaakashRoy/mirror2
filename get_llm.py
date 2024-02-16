import json
import requests

stream = False

url = "https://chat.tune.app/api/chat/completions"
headers = {
    "Authorization": "tune-b4042fc3-b3ae-4b05-a24e-b26dc3b2c0241708053579",
    "Content-Type": "application/json"
}


def ai_out(query):
    data = {
        "temperature": 0.5,
        "messages": [
            {
                "role": "system",
                "content": "You are assistant of an ai company named 'mirror2', here to help anyone with their personal documents"
            },
            {
                "role": "user",
                "content": "Act like, you're the assistant smart, genuine person"
            }
        ],
        "model": "mixtral-8x7b-inst-v0-1-32k",
        "stream": stream,
        "max_tokens": 300
    }
    response =  requests.post(url, headers=headers, json=data).json()
    response = response['choices'][0]['message']
    return response ['content']