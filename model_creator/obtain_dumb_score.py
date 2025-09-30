
from model_creator.misc import *
import shutil

initial_size = 224
bottleneck_size = 46

resize_all_images("comparison_images", "comparison_images_temp", bottleneck_size)
resize_all_images("comparison_images_temp", "comparison_images_temp_temp", initial_size)

image_differences = compute_image_differences("comparison_images", "comparison_images_temp_temp")
total_difference = sum(difference / 255.0 for difference in image_differences.values())

average_difference = calculate_average_difference_percentage(
        total_difference,
        len(image_differences),
        initial_size,
        initial_size,
)

if average_difference is not None:
        print('===> average difference per pixel ', round(average_difference, 2))

for temp_dir_name in ("comparison_images_temp", "comparison_images_temp_temp"):
    temp_dir = Path(temp_dir_name)
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
        