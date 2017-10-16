from simple_linear_model import *
import cv2

def read_and_display(image=None, is_display=False):
    if image is None:
        # Read default image file
        img = cv2.imread('digits.jpg', 0)
    else:
        img = image

    if not is_display:
        return img

    # Display an image
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
    k = cv2.waitKey(0)

    if k == 27:
        cv2.destroyAllWindows()

    return img


# raw = data.train.images[0]
# print data.train.images[0]
# data2 = np.reshape(raw, (28, 28))
# read_and_display(data2, True)
# print raw


# Image with digits from 1 to 9
digits_image = read_and_display()

# Blank 28x28 image
blank_image = cv2.imread('seven_font.png', 0)
roi = digits_image[0:460, 250:480]

# Resize image to 28 x 28
# height, width = blank_image.shape[:2]
# if height > width:
#     width = width * 28 / height
#     height = 28
# else:
#     height = height * 28 / width
#     width = 28
blank_image = cv2.resize(blank_image, (28, 28), interpolation=cv2.INTER_LINEAR)

blank_image = cv2.bitwise_not(blank_image)
# blank_image = blank_image * 1.0 / 255

data2 = np.reshape(blank_image, (-1, 784))
print data2

# Run tensorflow's session
run_session()
training()

# Predict images
predict_image(data2)
read_and_display(blank_image, is_display=True)


# Validation Set
# data3 = data.validation.images[125:126]
# data3_image = np.reshape(data3, (28, 28))
# predict_image(data3)
# read_and_display(data3_image, is_display=True)