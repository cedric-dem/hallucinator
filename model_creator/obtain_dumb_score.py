from pathlib import Path
from PIL import Image
import shutil

def resize_all_images(old_directory, new_directory, new_size):
    old_dir = Path(old_directory)
    new_dir = Path(new_directory)
    
    new_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(new_size, int):
        target_size = (new_size, new_size)
    elif isinstance(new_size, (tuple, list)) and len(new_size) == 2:
        target_size = tuple(new_size)
    else:
        raise ValueError("new_size not ok")

    valid_extensions = {".png", ".jpg", ".jpeg"}

    for file in old_dir.iterdir():
        if not file.is_file() or file.suffix.lower() not in valid_extensions:
            continue

        try:
            img = Image.open(file)
            img = img.resize(target_size, Image.LANCZOS)
            out_path = new_dir / file.name
            ext = file.suffix.lower()

            if ext in {".jpg", ".jpeg"}:
                if img.mode in {"RGBA", "P"}:
                    img = img.convert("RGB")
                img.save(out_path, quality=95, optimize=True)
            elif ext == ".png":
                if img.mode == "P":
                    img = img.convert("RGBA")
                img.save(out_path, optimize=True)
            else:
                img.save(out_path)
        except Exception as e:
            print(f"Error {file.name} : {e}")


def compute_image_differences(original_directory, processed_directory):
    original_dir = Path(original_directory)
    processed_dir = Path(processed_directory)

    differences = {}
    for processed_file in processed_dir.iterdir():
        if not processed_file.is_file():
            continue

        original_file = original_dir / processed_file.name
        if not original_file.exists():
            continue

        try:
            with Image.open(original_file) as original_img, Image.open(
                processed_file
            ) as processed_img:
                original_rgb = original_img.convert("RGB")
                processed_rgb = processed_img.convert("RGB")

                if original_rgb.size != processed_rgb.size:
                    original_rgb = original_rgb.resize(processed_rgb.size, Image.LANCZOS)

                diff_total = 0
                for orig_pixel, proc_pixel in zip(
                    original_rgb.getdata(), processed_rgb.getdata()
                ):
                    diff_total += sum(abs(o - p) for o, p in zip(orig_pixel, proc_pixel))

                differences[processed_file.name] = diff_total
        except Exception as e:
            print(f"Error comparing {processed_file.name}: {e}")

    return differences


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
        