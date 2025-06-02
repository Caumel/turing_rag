import requests

files = {'file': open('image.jpg', 'rb')}
response = requests.post("http://localhost:8000/detect", files=files)
print(response.json())
