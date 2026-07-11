import os
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    load_img,
    img_to_array
)


def generate_images(
    image_path,
    output_dir,
    rotation=40,
    width_shift=0.4,
    height_shift=0.4,
    shear=0.4,
    zoom=0.4,
    horizontal_flip=True,
    count=10
):
    """
    Generate augmented images from a single uploaded image.
    """

    # Create output folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Clear old generated images
    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Image augmentation configuration
    datagen = ImageDataGenerator(
        rotation_range=rotation,
        width_shift_range=width_shift,
        height_shift_range=height_shift,
        shear_range=shear,
        zoom_range=zoom,
        horizontal_flip=horizontal_flip,
        fill_mode="nearest"
    )

    # Load uploaded image
    img = load_img(image_path)

    x = img_to_array(img)

    x = x.reshape((1,) + x.shape)

    generated_images = []

    i = 0

    # Generate augmented images
    for _ in datagen.flow(
        x,
        batch_size=1,
        save_to_dir=output_dir,
        save_prefix="aug",
        save_format="jpeg"
    ):

        i += 1

        latest_file = max(
            [
                os.path.join(output_dir, f)
                for f in os.listdir(output_dir)
            ],
            key=os.path.getctime
        )

        generated_images.append(os.path.basename(latest_file))

        if i >= count:
            break

    return generated_images