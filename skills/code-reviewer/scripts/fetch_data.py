import requests
import json

def fetch_data(url):
    response = requests.get(url)
    return response.json()

if __name__ == '__main__':
    print(fetch_data('https://api.example.com/data'))
