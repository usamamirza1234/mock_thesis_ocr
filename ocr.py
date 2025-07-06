import cv2
from PIL import Image


im_file = "data/page_01.jpg"

im = Image.open(im_file)

print(im)

# im.show()
im.rotate(90)

im.save("temp/page_01.jpg")