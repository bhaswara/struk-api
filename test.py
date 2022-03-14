import requests
import time
API_URL = "http://localhost:5000/detect"

IMAGE_PATH = "test_images/struk-png-1.png"
#URL_IMAGE = "https://s3-jaki.jakarta.go.id/jaki/report/media/bb1f5543-5bba-453c-8b12-7cb2ec2a7af2"

'''
# Using URL
# GET request
payload = {'url': URL_IMAGE}
r = requests.get(API_URL, params=payload)
print("status code: {}".format(r.status_code))
print("headers: {}".format(r.headers))
print("content: {}".format(r.json()))
'''

#'''
# Using Image
# POST request
#start = time.time()
image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}
r = requests.post(API_URL, files=payload)
print("status code: {}".format(r.status_code))
print("headers: {}".format(r.headers))
print("content: {}".format(r.json()))
#end = time.time()
#elapsed = end-start
#print(elapsed)
#'''