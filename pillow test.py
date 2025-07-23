from PIL import Image
import numpy as np
import PIL.ImageOps

image_path = "C:\\Users\\alexa\\Pictures\\conv_image.png"
pil_image = Image.open(image_path)
grayImage = pil_image.convert("L")
invImage = PIL.ImageOps.invert(grayImage)


x,y = invImage.size
for j in range(x):
    for i in range(y):
        pixel=invImage.getpixel((i,j))
        
        print(pixel, end=" ")
    print("\n")





