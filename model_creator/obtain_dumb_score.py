
from pathlib import Path
from PIL import Image

INPUT_DIR = Path("comparison_images")
OUTPUT_DIR = Path("comparison_images_temp")
TARGET_SIZE = 40

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for file in INPUT_DIR.iterdir():
    if not file.is_file():
        continue
    if file.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        continue
    try:
        img = Image.open(file)
        img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
        ext = file.suffix.lower()
        out_path = OUTPUT_DIR / file.name
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
    except Exception:
        pass
