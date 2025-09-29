from pathlib import Path
from PIL import Image

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


resize_all_images("comparison_images", "comparison_images_temp", 40)
resize_all_images("comparison_images_temp", "comparison_images_temp_temp", 224)
