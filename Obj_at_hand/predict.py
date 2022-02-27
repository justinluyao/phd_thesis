#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
from frcnn import FRCNN
from PIL import Image

frcnn = FRCNN()

# while True:
    # img = input('Input image filename:')
try:
    image = Image.open('H:\object_detection_images、test\P01\P01_11/0000000301.jpg')
    image=image.resize((1280,720))
except:
    print('Open Error! Try again!')
else:
    r_image = frcnn.detect_image(image)
    r_image.show()
