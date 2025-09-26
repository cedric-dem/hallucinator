from PIL import Image

def add_images(image1_path, image2_path):
    img1 = Image.open(image1_path).convert("RGB")
    img2 = Image.open(image2_path).convert("RGB")

    # Ensure both images have the same size
    if img1.size != img2.size:
        raise ValueError("The images must have the same dimensions.")

    # Get pixel data
    pixels1 = img1.load()
    pixels2 = img2.load()

    total = 0

    for y in range(img1.height):
        for x in range(img1.width):
            r1, g1, b1 = pixels1[x, y]
            r2, g2, b2 = pixels2[x, y]

            r_sum = abs(r1 - r2)
            g_sum = abs(g1 - g2)
            b_sum = abs(b1 - b2)

            total += r_sum + g_sum +  b_sum

    dpcpp = total / (img1.width * img1.height * 3 )

    print("===> avg delta per channel per pixel",round(dpcpp,2))

print("==> reduced by hand")
add_images("_img_in.jpg", "img_redid.jpg")

print("==> reduced by nn")
add_images("_img_in.jpg", "270.png")