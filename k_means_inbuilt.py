import numpy as np
import cv2

img = cv2.imread('hept.jpg')
Z = img.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 5
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

cv2.imshow('res2',res2)
cv2.waitKey(0)
cv2.destroyAllWindows()


center.tolist()
j = [i[::-1] for i in center] 
print(j)
resultBGR=[]
cnt=0
for i in j:
    
    bgr = i
    thresh = 40
 
    minBGR = np.array([bgr[0] - thresh, bgr[1] - thresh, bgr[2] - thresh])
    maxBGR = np.array([bgr[0] + thresh, bgr[1] + thresh, bgr[2] + thresh])
    print(minBGR,maxBGR)
 
    maskBGR = cv2.inRange(res2,minBGR,maxBGR)
    resultBGR += [cv2.bitwise_and(res2, res2, mask = maskBGR)]
    #cv2.imshow("Result BGR"+str(cnt), resultBGR[cnt])
    cv2.imshow("",maskBGR)
    cv2.waitKey()
    cv2.destroyAllWindows()
    cnt+=1

