import cv2
import numpy as np
# import matplotlib.pyplot as plt


def erosion(img, kernel):

    (l,b) = img.shape
    p1 = kernel.shape[0] // 2
    p2 = kernel.shape[1] // 2

    img1 = np.zeros((l + 2*p1 ,b + 2*p2), dtype = np.uint8)
    img1[p1:p1+l, p2:p2+b] = img
    final_image = np.zeros((l,b))
    final_image = final_image.astype(np.uint8)
    kernel = kernel//255

    for i in range(l):
        for j in range(b):
            y = i + p1
            x = j + p2
            window = img1[y-p1:y+p1+1, x-p2:x+p2+1]
            bin_window = window // 255
            mix = np.bitwise_and(bin_window, kernel)
            if (mix == kernel).all():
                final_image[i,j] = 255

    return final_image


def dilation(img,kernel):

    (l,b) = img.shape
    p1 = kernel.shape[0] // 2
    p2 = kernel.shape[1] // 2

    img1 = np.zeros((l + 2*p1 ,b + 2*p2), dtype = np.uint8)
    img1[p1:p1+l, p2:p2+b] = img
    final_image = np.zeros((l,b))
    final_image = final_image.astype(np.uint8)
    kernel = kernel//255

    for i in range(l):
        for j in range(b):
            y = i + p1
            x = j + p2
            window = img1[y-p1:y+p1+1, x-p2:x+p2+1]
            bin_window = window // 255
            mix = np.bitwise_and(bin_window, kernel)
            if (np.sum(mix)>0):
                final_image[i,j] = 255

    return final_image


def opening(img,kernel):
    return dilation(erosion(img, kernel), kernel)


def closing(img,kernel):
    return erosion(dilation(img, kernel), kernel)


# Found in jupyter-notebook
answer_key = {1: 2, 2: 3,  3: 1,  4: 1,  5: 4,  6: 1,  7: 3,  8: 3,  9: 1,  10: 3,  11: 1,  12: 2,  13: 3,  14: 3,  15: 2,  16: 1,  17: 4,  18: 2,  19: 3,  20: 2,  21: 4,  22: 3,  23: 4,  24: 2,  25: 4,  26: 3,  27: 4,  28: 4,  29: 2,  30: 3,  31: 2,  32: 2,  33: 4,  34: 3,  35: 2,  36: 3,  37: 2,  38: 3,  39: 3,  40: 1,  41: 2,  42: 2,  43: 3,  44: 3,  45: 2}
cart_prod = np.array([[( 60,  83), ( 60, 125), ( 60, 167), ( 60, 209)], [(103,  83), (103, 125), (103, 167), (103, 209)], [(145,  83), (145, 125), (145, 167), (145, 209)], [(187,  83), (187, 125), (187, 167), (187, 209)], [(230,  83), (230, 125), (230, 167), (230, 209)], [(272,  83), (272, 125), (272, 167), (272, 209)], [(314,  83), (314, 125), (314, 167), (314, 209)], [(356,  83), (356, 125), (356, 167), (356, 209)], [(399,  83), (399, 125), (399, 167), (399, 209)], [(441,  83), (441, 125), (441, 167), (441, 209)], [(483,  83), (483, 125), (483, 167), (483, 209)], [(526,  83), (526, 125), (526, 167), (526, 209)], [(568,  83), (568, 125), (568, 167), (568, 209)], [(610,  83), (610, 125), (610, 167), (610, 209)], [(653,  83), (653, 125), (653, 167), (653, 209)], [( 60, 420), ( 60, 462), ( 60, 504), ( 60, 546)], [(103, 420), (103, 462), (103, 504), (103, 546)], [(145, 420), (145, 462), (145, 504), (145, 546)], [(187, 420), (187, 462), (187, 504), (187, 546)], [(230, 420), (230, 462), (230, 504), (230, 546)], [(272, 420), (272, 462), (272, 504), (272, 546)], [(314, 420), (314, 462), (314, 504), (314, 546)], [(356, 420), (356, 462), (356, 504), (356, 546)], [(399, 420), (399, 462), (399, 504), (399, 546)], [(441, 420), (441, 462), (441, 504), (441, 546)], [(483, 420), (483, 462), (483, 504), (483, 546)], [(526, 420), (526, 462), (526, 504), (526, 546)], [(568, 420), (568, 462), (568, 504), (568, 546)], [(610, 420), (610, 462), (610, 504), (610, 546)], [(653, 420), (653, 462), (653, 504), (653, 546)], [( 60, 757), ( 60, 799), ( 60, 841), ( 60, 883)], [(103, 757), (103, 799), (103, 841), (103, 883)], [(145, 757), (145, 799), (145, 841), (145, 883)], [(187, 757), (187, 799), (187, 841), (187, 883)], [(230, 757), (230, 799), (230, 841), (230, 883)], [(272, 757), (272, 799), (272, 841), (272, 883)], [(314, 757), (314, 799), (314, 841), (314, 883)], [(356, 757), (356, 799), (356, 841), (356, 883)], [(399, 757), (399, 799), (399, 841), (399, 883)], [(441, 757), (441, 799), (441, 841), (441, 883)], [(483, 757), (483, 799), (483, 841), (483, 883)], [(526, 757), (526, 799), (526, 841), (526, 883)], [(568, 757), (568, 799), (568, 841), (568, 883)], [(610, 757), (610, 799), (610, 841), (610, 883)], [(653, 757), (653, 799), (653, 841), (653, 883)]], dtype=[('f0', '<i4'), ('f1', '<i4')])

hoe1 = np.full((25, 25, 3), 255)
for x in range(25):
    for y in range(25):
        if (x-12)**2 + (y-12)**2 < 100:
            hoe1[x,y,:] = 0
        elif (x-12)**2 + (y-12)**2 < 170:
            hoe1[x,y,:] = -1
hoe1 = hoe1[:,:,0]

t = int(input())
for _ in range(t):

    path_to_img = input()

    # Reading the image
    img = cv2.imread(path_to_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[:,:,0]
    img = img>190
    img = img.astype(int)
    img = img*255
    demn = img[740:1440, 140:1060]

    # Opening the image
    kernel = 255*np.ones((7, 7),dtype = np.uint8)
    demn = opening(np.copy(demn), kernel)
    # plt.imshow(demn, cmap='gray')
    # plt.show()

    # Locating the circles
    [d0, d1] = demn.shape
    [f0, f1] = hoe1.shape
    ans = []
    for x in range(d0):
        for y in range(d1):
            if f0+x-1<d0 and f1+y-1<d1:
                flg=0
                for xx in range(f0):
                    for yy in range(f1):
                        if hoe1[xx,yy]==-1:
                            continue
                        elif hoe1[xx,yy]!=demn[x+xx, y+yy]:
                            flg=1
                            break
                if flg==0:
                    ans.append([x, y])
    ans = sorted(ans, key = ans.count, reverse = True)

    # Pruning the redundant circles
    gg = []
    for i in ans:
        flg=0
        for j in gg:
            if abs(i[0]-j[0])<=10 and abs(i[1]-j[1])<=10:
                flg=1
        if flg==0:
            gg.append(i)

    # Matching with the answer key
    marked = {}
    for i in range(1, 46): marked[i]=-1
    def qna(spot):
        global marked
        qn  = -1
        opt = -1
        cur = 1e5
        for idx1, i in enumerate(cart_prod):
            for idx2, j in enumerate(i):
                if abs(j[0]-spot[0])**2+abs(j[1]-spot[1])**2 < cur:
                    qn=idx1+1
                    opt=idx2+1
                    cur = abs(j[0]-spot[0])**2 + abs(j[1]-spot[1])**2
        marked[qn] = opt    
    for i in gg:
        qna(i)

    # # Printing the answers marked
    for i in range(1, 46):
        if marked[i]==1: print('A')
        elif marked[i]==2: print('B')
        elif marked[i]==3: print('C')
        elif marked[i]==4: print('D')
        else: print(-1)
