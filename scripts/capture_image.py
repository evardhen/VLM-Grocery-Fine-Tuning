import requests
import os
import re
import time

# URL to capture the image from
url_capture = "http://kitchen_cabinet_4/capture"
url_capture_with_led = "http://kitchen_cabinet_4/capture_with_led"
# Directory where images will be saved
dir_path = "datasets/binaries/captured_images_home/"
os.makedirs(dir_path, exist_ok=True)

# Find the current maximum index of the captured images
files = os.listdir(dir_path)
pattern = re.compile(r"captured_image_(\d+)\.jpg") 
max_index = -1
for f in files:
    match = pattern.match(f)
    if match:
        idx = int(match.group(1))
        if idx > max_index:
            max_index = idx

next_index = max_index + 1
filename = f"captured_image_{next_index}.jpg"
file_path = os.path.join(dir_path, filename)

# Retrieve the image
try:
    response = requests.get(url_capture, stream=True)
    response.raise_for_status()

    # Save the image
    with open(file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Image successfully saved as {file_path}")
except requests.exceptions.RequestException as e:
    print(f"Failed to retrieve image: {e}")

# time.sleep(0.5)
# next_index = next_index + 1
# filename = f"captured_image_{next_index}.jpg"
# file_path = os.path.join(dir_path, filename)

# try:
#     response = requests.get(url_capture_with_led, stream=True)
#     response.raise_for_status()

#     # Save the image
#     with open(file_path, "wb") as f:
#         for chunk in response.iter_content(chunk_size=8192):
#             f.write(chunk)

#     print(f"Image successfully saved as {file_path}")
# except requests.exceptions.RequestException as e:
#     print(f"Failed to retrieve image: {e}")