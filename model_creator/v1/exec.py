# deepdream_min.py
import numpy as np, PIL.Image as Image, tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import img_to_array

def load_img(path, max_dim=800):
    img = Image.open(path).convert("RGB")
    scale = max_dim / max(img.size)
    img = img.resize((int(img.width*scale), int(img.height*scale)), Image.LANCZOS)
    arr = img_to_array(img)
    return tf.convert_to_tensor(arr[None, ...] / 255.0)

def unprocess(img):
    img = tf.clip_by_value(img[0], 0.0, 1.0)
    img = tf.image.convert_image_dtype(img, dtype=tf.uint8)
    return Image.fromarray(img.numpy())

def make_deepdream_model(base):
    layer_names = [
        "mixed3",
    ]
    layers = [base.get_layer(name).output for name in layer_names]
    return tf.keras.Model(inputs=base.input, outputs=layers)

def calc_loss(model, img):
    # Somme des énergies des features (L2)
    activations = model(img)
    if not isinstance(activations, list): activations = [activations]
    losses = [tf.reduce_mean(tf.square(act)) for act in activations]
    return tf.add_n(losses)

@tf.function
def deepdream_step(model, img, step_size):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = calc_loss(model, img)
    grads = tape.gradient(loss, img)
    grads = grads / (tf.math.reduce_std(grads) + 1e-8)
    img = img + step_size * grads
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, loss

def deepdream(img, model, steps=100, step_size=0.01):
    for _ in range(steps):
        img, loss = deepdream_step(model, img, step_size)
    return img

if __name__ == "__main__":
    base = load_model("v1/model.keras", compile=False)
    dreamer = make_deepdream_model(base)

    img = load_img("v1/_img_in.jpg")

    out = deepdream(img, dreamer, steps=80, step_size=0.01)

    unprocess(out).save("v1/_img_out.jpg")
    print("OK")