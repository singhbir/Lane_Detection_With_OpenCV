import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_coodenates(image,line_parameters):
    slope,intercept=line_parameters
    y1=image.shape[0]
    y2=int(y1*(3/5))
    x1=int((y1-intercept)/slope) # y=mx+b   x=(y-b)/m 
    x2=int((y2-intercept)/slope) # y=mx+b   x=(y-b)/m 
    return np.array([x1,y1,x2,y2])

def average_slope_intercept(image,lines):
    #when x increase with increase in y its slope is +ve and when x decrease with increase in y or vise-versa slope is -ve
    left_fit=[]
    right_fit=[]
    for line in lines:
        x1,y1,x2,y2=line.reshape(4)
        parameters=np.polyfit((x1,x2),(y1,y2),1) 
        slope=parameters[0]
        intercept=parameters[1] #when x increase with increase in y its slope is +ve and when x decrease with increase in y or vise-versa slope is -ve
        if slope < 0 :
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    left_fit_average=np.average(left_fit,axis=0)
    right_fit_average=np.average(right_fit,axis=0)
    left_line = make_coodenates(image,left_fit_average)
    right_line = make_coodenates(image,right_fit_average)
    return np.array([left_line,right_line]) 


    # print(left_fit)
    # print(right_fit)

 
def canny(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #we are converting it into gray scale beacuse it is faster to process one channel rather than 3 channels(rgb)
    blur=cv2.GaussianBlur(gray,(5,5),0)
    canny=cv2.Canny(blur,50,150) #we generally use one : three and canny is used to detect sudden change in gradient and find the edges
    return canny

def region_of_interest(image):
    height=image.shape[0]
    polygon=np.array([
        [(200,height),(1100,height),(550,250)]])
    mask=np.zeros_like(image)
    cv2.fillPoly(mask,polygon,255)
    masked_image=cv2.bitwise_and(image,mask)
    return masked_image

def display_line(image,lines):
    line_image=np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2=line.reshape(4)
            cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),10)
    return line_image

# image=cv2.imread('test_image.jpg')
# lane_image=np.copy(image)
# canny = canny(lane_image)
# crop_image=region_of_interest(canny)
# lines=cv2.HoughLinesP(crop_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)

# averaged_lines=average_slope_intercept(lane_image,lines)
# line_image=display_line(lane_image,averaged_lines)
# combo_image=cv2.addWeighted(lane_image,0.8,line_image,1,1)
# cv2.imshow("image",combo_image)
# cv2.waitKey(0) 

cap=cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    rate,frame=cap.read()
    canny_img = canny(frame)
    crop_image=region_of_interest(canny_img)
    lines=cv2.HoughLinesP(crop_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
    averaged_lines=average_slope_intercept(frame,lines)
    line_image=display_line(frame,averaged_lines)
    combo_image=cv2.addWeighted(frame,0.8,line_image,1,1)
    cv2.imshow("image",combo_image)
    cv2.waitKey(1)  




