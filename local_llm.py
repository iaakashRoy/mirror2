import requests
import json

url = "http://localhost:11434/api/generate"

headers = {
    'Content-Type': 'application/json'  # corrected syntax
}

def get_answer(prompt):

    data = {
    "model": "tinydolphin",
    "stream": False,
    "prompt": prompt
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        raw_text = json.loads(response.text)
        actual_response = raw_text["response"]
        return actual_response
    else:
        return "Error:llm_response"
