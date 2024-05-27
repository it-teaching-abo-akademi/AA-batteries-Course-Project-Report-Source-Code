import cv2
import numpy as np
import sys

def preprocess_image(image_path:str, target_size=(28, 28))->np.ndarray:
    """reads and preprocesses the image by resizing
    and converting to grayscale."""
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def save_to_header(data:np.ndarray, output_file:str):
    """Flattens the image array in row-major order
    and writes it as a C header file in int32 array format."""
    with open(output_file, 'w') as f:
        f.write('#ifndef IMAGE_DATA_H\n')
        f.write('#define IMAGE_DATA_H\n\n')
        f.write('#include <stdint.h>\n\n')
        f.write('const int image_data[] = {\n')

        for row in data:
            for value in row:
                f.write(f'{value}, ')
            f.write('\n')

        f.write('};\n\n')
        f.write('#endif // IMAGE_DATA_H\n')

if __name__ == "__main__":
    # Todo: Add path for image/images and the output file name with .h extension (eg- filename.h)
    image_path = str(sys.argv[1])
    output_file = str(sys.argv[2])

    processed_image = preprocess_image(image_path)
    save_to_header(processed_image, output_file)
