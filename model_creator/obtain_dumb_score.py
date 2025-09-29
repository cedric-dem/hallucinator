
from model_creator.misc import *
import shutil

initial_size = 224
bottleneck_size = 40

resize_all_images("comparison_images", "comparison_images_temp", bottleneck_size)
resize_all_images("comparison_images_temp", "comparison_images_temp_temp", initial_size)

image_differences = compute_image_differences("comparison_images", "comparison_images_temp_temp")
all_sum = 0

for image_name, difference in image_differences.items():
	all_sum+=difference

delta = all_sum / (len(image_differences) * 3 *  initial_size *  initial_size)

print('===> average difference per pixel ',round((delta )* 100,2)) #*255 ? todo isolate with main


for temp_dir_name in ("comparison_images_temp", "comparison_images_temp_temp"):
    temp_dir = Path(temp_dir_name)
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
        