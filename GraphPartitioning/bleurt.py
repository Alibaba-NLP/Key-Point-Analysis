import requests

url = "http://localhost:5000/score"
headers = {
    'Content-Type': 'application/json'
}


def bleurt(references, candidates):
    data = {"references": references, "candidates": candidates}
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        scores = response.json()
        return scores
    else:
        print(response)
        print(f'error to request bleurt: {response.status_code}')
        exit(-1)
