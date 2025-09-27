import os
from PIL import Image

directory = "flickr/224"

target_size = (224, 224)
extensions = ('.png', '.jpg', '.jpeg')

for filename in os.listdir(directory):
	if filename.lower().endswith(extensions):
		filepath = os.path.join(directory, filename)

		try:
			with Image.open(filepath) as img:
				img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
				img_resized.save(filepath)

		except Exception as e:
			print(f"Error with {filename} : {e}")

print("done")
