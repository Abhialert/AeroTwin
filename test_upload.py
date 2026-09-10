import requests

url = 'http://127.0.0.1:5000/api/upload'
video_path = r'C:\Users\ABHISHEK\Downloads\df_cngndpp.mp4'

with open(video_path, 'rb') as f:
    files = {'video': ('df_cngndpp.mp4', f, 'video/mp4')}
    response = requests.post(url, files=files)

print("Status Code:", response.status_code)
print("Response:", response.text)
