from tensorflow.keras.preprocessing import image as keras_image
import numpy as np

def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")
    # resize the input image and preprocess it
    image = image.resize(target)
    image = keras_image.img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # return the processed image
    return image
