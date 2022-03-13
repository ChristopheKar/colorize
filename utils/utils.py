import cv2


def save_image(image, output_path, to_bgr=True):

    """Write image to disk using OpenCV."""

    # Convert image to uint8 if needed
    if ('float' in image.dtype.name):
        image = (image*255).astype('uint8')

    # Convert image from RGB to BGR if needed
    if (to_bgr):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Save image
    cv2.imwrite(output_path, image)