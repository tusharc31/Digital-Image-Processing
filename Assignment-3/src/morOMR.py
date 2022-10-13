# Author: @BVK
# Assignment 3: Question 1 Boiler Plate
import cv2 as cv
import numpy as np
# do not import any other library
# Note: this is just a boiler plate
# feel free to make changes in the structure
# however, input/output should essentially be the same.
# IMPORTANT: When you use this, save it in the path src/morOMR.py - if you don't
# your test will fail automatically.


#############################################
#############################################
### GLOBAL UTILITIES

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
answer_key = {1: 2, 2: 3, 3: 1, 4: 1, 5: 4, 6: 1, 7: 3, 8: 3, 9: 1, 10: 3, 11: 1, 12: 2, 13: 3, 14: 3, 15: 2, 16: 1, 17: 4, 18: 2, 19: 3, 20: 2, 21: 4, 22: 3, 23: 4, 24: 2, 25: 4, 26: 3, 27: 4, 28: 4, 29: 2, 30: 3, 31: 2, 32: 2, 33: 4, 34: 3, 35: 2, 36: 3, 37: 2, 38: 3, 39: 3, 40: 1, 41: 2, 42: 2, 43: 3, 44: 3, 45: 2}
cart_prod = np.array([[( 20,  23), ( 20,  65), ( 20, 107), ( 20, 149)],
       [( 63,  23), ( 63,  65), ( 63, 107), ( 63, 149)],
       [(105,  23), (105,  65), (105, 107), (105, 149)],
       [(147,  23), (147,  65), (147, 107), (147, 149)],
       [(190,  23), (190,  65), (190, 107), (190, 149)],
       [(232,  23), (232,  65), (232, 107), (232, 149)],
       [(274,  23), (274,  65), (274, 107), (274, 149)],
       [(316,  23), (316,  65), (316, 107), (316, 149)],
       [(359,  23), (359,  65), (359, 107), (359, 149)],
       [(401,  23), (401,  65), (401, 107), (401, 149)],
       [(443,  23), (443,  65), (443, 107), (443, 149)],
       [(486,  23), (486,  65), (486, 107), (486, 149)],
       [(528,  23), (528,  65), (528, 107), (528, 149)],
       [(570,  23), (570,  65), (570, 107), (570, 149)],
       [(613,  23), (613,  65), (613, 107), (613, 149)],
       [( 20, 360), ( 20, 402), ( 20, 444), ( 20, 486)],
       [( 63, 360), ( 63, 402), ( 63, 444), ( 63, 486)],
       [(105, 360), (105, 402), (105, 444), (105, 486)],
       [(147, 360), (147, 402), (147, 444), (147, 486)],
       [(190, 360), (190, 402), (190, 444), (190, 486)],
       [(232, 360), (232, 402), (232, 444), (232, 486)],
       [(274, 360), (274, 402), (274, 444), (274, 486)],
       [(316, 360), (316, 402), (316, 444), (316, 486)],
       [(359, 360), (359, 402), (359, 444), (359, 486)],
       [(401, 360), (401, 402), (401, 444), (401, 486)],
       [(443, 360), (443, 402), (443, 444), (443, 486)],
       [(486, 360), (486, 402), (486, 444), (486, 486)],
       [(528, 360), (528, 402), (528, 444), (528, 486)],
       [(570, 360), (570, 402), (570, 444), (570, 486)],
       [(613, 360), (613, 402), (613, 444), (613, 486)],
       [( 20, 697), ( 20, 739), ( 20, 781), ( 20, 823)],
       [( 63, 697), ( 63, 739), ( 63, 781), ( 63, 823)],
       [(105, 697), (105, 739), (105, 781), (105, 823)],
       [(147, 697), (147, 739), (147, 781), (147, 823)],
       [(190, 697), (190, 739), (190, 781), (190, 823)],
       [(232, 697), (232, 739), (232, 781), (232, 823)],
       [(274, 697), (274, 739), (274, 781), (274, 823)],
       [(316, 697), (316, 739), (316, 781), (316, 823)],
       [(359, 697), (359, 739), (359, 781), (359, 823)],
       [(401, 697), (401, 739), (401, 781), (401, 823)],
       [(443, 697), (443, 739), (443, 781), (443, 823)],
       [(486, 697), (486, 739), (486, 781), (486, 823)],
       [(528, 697), (528, 739), (528, 781), (528, 823)],
       [(570, 697), (570, 739), (570, 781), (570, 823)],
       [(613, 697), (613, 739), (613, 781), (613, 823)]],
      dtype=[('f0', '<i4'), ('f1', '<i4')])

structuring_element = np.full((25, 25, 3), 255)
for x in range(25):
    for y in range(25):
        if (x-12)**2 + (y-12)**2 < 81:
            structuring_element[x,y,:] = 0
        elif (x-12)**2 + (y-12)**2 < 170:
            structuring_element[x,y,:] = -1
structuring_element = structuring_element[:,:,0]

#########################################################
#########################################################


def getAnswers(img)->list:

    img = img[:,:,0]
    img = img>190
    img = img.astype(int)
    img = img*255
    cropped_img = img[780:1440, 200:1060]
    # plt.imshow(cropped_img, cmap='gray')
    # plt.show()
    

    # Opening the image
    kernel = 255*np.ones((3, 3),dtype = np.uint8)
    cropped_img = opening(np.copy(cropped_img), kernel)
    # plt.imshow(cropped_img, cmap='gray')
    # plt.show()
    # return

    # Locating the circles
    [d0, d1] = cropped_img.shape
    [f0, f1] = structuring_element.shape
    ans = []
    for x in range(d0):
        for y in range(d1):
            if f0+x-1<d0 and f1+y-1<d1:
                flg=0
                for xx in range(f0):
                    for yy in range(f1):
                        if structuring_element[xx,yy]==-1:
                            continue
                        elif structuring_element[xx,yy]!=cropped_img[x+xx, y+yy]:
                            flg=1
                            break
                if flg==0:
                    ans.append([x, y])
    ans = sorted(ans, key = ans.count, reverse = True)

    # Pruning the redundant circles
    ans_pruned = []
    for i in ans:
        flg=0
        for j in ans_pruned:
            if abs(i[0]-j[0])<=10 and abs(i[1]-j[1])<=10:
                flg=1
        if flg==0:
            ans_pruned.append(i)

    # Matching with the answer key
    marked = {}
    for i in range(1, 46): marked[i]=-1
    def qna(spot):
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
    for i in ans_pruned:
        qna(i)

    answers = []

    # Printing the answers marked
    for i in range(1, 46):
        if marked[i]==1: answers.append('A')
        elif marked[i]==2: answers.append('B')
        elif marked[i]==3: answers.append('C')
        elif marked[i]==4: answers.append('D')
        else: answers.append('-1')

    return answers


if __name__ == "__main__":
    
    # Read the number of test cases
    # input() returns str by default, i.e. 1000 is read as '1000'.
    # .strip() used here to strip of the trailing `\n` character
      
    T = int(input().strip())
                            
    for i in range(T):
        
        fileName = input().strip() # read path to image
        omr_sheet = cv.imread(fileName)
        
        answers = getAnswers(omr_sheet) # fetch your answer

        for answer in answers: # assuming answers is a list
            print(answer)  # print() function automatically appends the `\n`