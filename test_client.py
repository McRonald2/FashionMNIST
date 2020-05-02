import requests

"""
A piece of code to test our api.
Sending POST requests and getting a response for 2 sample images.
"""

resp = requests.post("http://127.0.0.1:5000/",
                     files={'file': open('./data/trouser.jpg', 'rb')})

print(resp.json())

resp = requests.post("http://127.0.0.1:5000/",
                     files={'file': open('./data/shirt.jpg', 'rb')})

print(resp.json())