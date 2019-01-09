import cv2
import numpy as np
import math
import string
import random
import imutils
 
if __name__ == '__main__':
    all_images = []
    n = 8
    for i in range(70):
        all_images.append( './Grass/a'+str(i) +'.jpg')
    #font Details
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(200):
        img_no = np.random.randint(0,len(all_images))
        img = cv2.imread(all_images[img_no])
        h,w,c = img.shape
        color_shape = []
        color_txt = []
        for j in range(c):
            color_shape.append(np.random.randint(255))
            color_txt.append(np.random.randint(255))
        a = int(min(h/10,w/10)*np.random.random())
        while a < min(h/20,w/20):
            a = int(min(h/10,w/10)*np.random.random())
        b = int(min(h/6,w/6)*np.random.random())
        while b-a < a/10:
            b = int(min(h/6,w/6)*np.random.random())
        shape_img = np.zeros((2*a,2*b,3), np.uint8)
        theta = 360/n
        pts = []
        for j in range(n):
            pts.append([int(b+min(a,b)*math.sin(j*theta*math.pi/180.0)),int(a-min(a,b)*math.cos(j*theta*math.pi/180.0))])
        cv2.fillConvexPoly(shape_img, np.array(pts), color_shape, 0)
        scale = 0.03*int(min(a,b))
        thickness = int(0.1*min(a,b))
        text = random.choice(string.ascii_letters)
        textSize =  cv2.getTextSize(text, font, scale, thickness)[0]
        (txt_x,txt_y) = (int(b-textSize[0]/2),int(a+textSize[1]/2))
        phi = np.random.randint(0,9)*45
        cv2.putText(shape_img,text,(txt_x,txt_y), font, scale,color_txt,thickness,cv2.LINE_AA,0)
        rotated = imutils.rotate(shape_img, phi)
        b = max(len(rotated[0]), len(rotated))
        (xleft, ytop) = (int((min(h,w)-b)*np.random.random()), int((min(h,w)-b)*np.random.random()))
        for x in range(len(rotated)):
            for y in range(len(rotated[0])):
                x1 = x + xleft 
                y1 = y + ytop
                if rotated[x, y, 0] != 0:
                    img[x1, y1] = rotated[x, y]
                    continue
                if rotated[x, y, 1] != 0:
                    img[x1, y1] = rotated[x, y]
                    continue
                if rotated[x, y, 2] != 0:
                    img[x1, y1] = rotated[x, y]
                    continue
        # cv2.imwrite('rotate' + str(i) + '.png', rotated)            
        cv2.imwrite('./Octagon/octagon' + str(i) + '.jpg', shape_img) 
