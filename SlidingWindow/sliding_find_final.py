#!/usr/bin/env python
# -*- coding: utf-8 -*-

from textwrap import wrap
import numpy as np
import cv2, random, math, copy
from matplotlib import pyplot as plt

Width = 640
Height = 480

#cap = cv2.VideoCapture("xycar_track1.mp4")
cap = cv2.VideoCapture("subProject.avi")
window_title = 'camera'

f = open("data.csv","w")

warp_img_w = 640
warp_img_h = 480

warpx_margin = 40
warpy_margin = 3

nwindows = 18
margin = 35
minpix = 10

lane_bin_th = 160
lane_max_th = 255

prev_l = 0
prev_r = 0
prev_mid_pos = 520
prev_l_pos = 0
prev_r_pos = 1040

#look closer bird eye view
warp_src  = np.array([
    [40-warpx_margin, 346-warpy_margin],  
    [40-warpx_margin, 396+warpy_margin],
    [600+warpx_margin, 346-warpy_margin],
    [600+warpx_margin, 396+warpy_margin]
], dtype=np.float32)
'''
warp_src  = np.array([
    [210-warpx_margin, 290-warpy_margin],  
    [40-warpx_margin, 390+warpy_margin],
    [430+warpx_margin, 290-warpy_margin],
    [600+warpx_margin, 390+warpy_margin]
], dtype=np.float32)
'''
#look closer bird eye view
warp_dist = np.array([
    [0,0],
    [0,warp_img_h],
    [warp_img_w,0],
    [warp_img_w, warp_img_h]
], dtype=np.float32)

calibrated = False
if calibrated:
    mtx = np.array([
        [422.037858, 0.0, 245.895397], 
        [0.0, 435.589734, 163.625535], 
        [0.0, 0.0, 1.0]
    ])
    dist = np.array([-0.289296, 0.061035, 0.001786, 0.015238, 0.0])
    cal_mtx, cal_roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (Width, Height), 1, (Width, Height))
else:
    mtx = np.array([
        [203.718327, 0.000000, 319.500000], 
        [0.000000, 203.718327, 239.500000], 
        [0.000000, 0.000000, 1.000000]
    ])
    dist = np.array([0.000000, 0.000000, 0.000000, 0.000000])
    cal_mtx, cal_roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (Width, Height), 1, (Width, Height))

def normalize_HLS_L(img):
    blur = cv2.GaussianBlur(img,(5, 5), 0)
    _, L, _ = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HLS))
    minVal, maxVal, MinLoc, maxLoc = cv2.minMaxLoc(L)
    L = (255 / maxVal) * L
    L = L.astype(np.uint8)
    
    lane = cv2.adaptiveThreshold(L, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 45, 21)
    
    lane = cv2.medianBlur(lane, 11)
    
    kernel = np.ones((5, 5), np.uint8)
    cv2.morphologyEx(lane, cv2.MORPH_OPEN, kernel)
    cv2.morphologyEx(lane, cv2.MORPH_CLOSE, kernel)
    cv2.morphologyEx(lane, cv2.MORPH_TOPHAT, kernel)
    
    return lane

def normalize_LAB_L(img):
    blur = cv2.GaussianBlur(img,(5, 5), 0)
    L, _, _ = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    minVal, maxVal, MinLoc, maxLoc = cv2.minMaxLoc(L)
    L = (255 / maxVal) * L
    L = L.astype(np.uint8)
    
    lane = cv2.adaptiveThreshold(L, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 45, 21)
    
    lane = cv2.medianBlur(lane, 11)
    
    kernel = np.ones((5, 5), np.uint8)
    cv2.morphologyEx(lane, cv2.MORPH_OPEN, kernel)
    cv2.morphologyEx(lane, cv2.MORPH_CLOSE, kernel)
    cv2.morphologyEx(lane, cv2.MORPH_TOPHAT, kernel)
    
    return lane
    
    

def calibrate_image(frame):
    '''
    global Width, Height
    global mtx, dist
    global cal_mtx, cal_roi
    tf_image = cv2.undistort(frame, mtx, dist, None, cal_mtx)
    x, y, w, h = cal_roi
    tf_image = tf_image[y:y+h, x:x+w]

    return cv2.resize(frame, (Width, Height))
    '''
    return frame

def warp_image(img, src, dst, size):
    #warp_image(image, warp_src, warp_dist, (warp_img_w, warp_img_h))
    newimg = cv2.bitwise_and(normalize_HLS_L(img), normalize_LAB_L(img))
    cv2.rectangle(newimg, (225, 399), (420, 480),(0, 0, 0), -1)
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    #warp_img = cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)
    warp_img = cv2.warpPerspective(newimg, M, size, flags=cv2.INTER_LINEAR)
    warp_img = cv2.cvtColor(warp_img, cv2.COLOR_GRAY2BGR)
    warp_img2 = cv2.warpPerspective(warp_img, Minv, (640, 480), flags=cv2.INTER_LINEAR)
    
    #cv2.imshow("warp_img", warp_img)
    #cv2.imshow("warp_img2", warp_img2)
    cv2.imshow("newimg", newimg)
    #_, warp_img = cv2.threshold(img, 100, lane_max_th, cv2.THRESH_BINARY)
    
    return warp_img, M, Minv

def warp_process_image(lane):
    global nwindows
    global margin
    global minpix
    global lane_bin_th
    global prev_l, prev_r, prev_l_pos, prev_r_pos
    
    lane = cv2.cvtColor(lane, cv2.COLOR_BGR2GRAY)
    extend_screen = np.zeros((480, 1040), np.uint8)
    extend_screen[0:0+480, 200:200+640] = lane
    lane = extend_screen
    
    histogram = np.sum(lane[lane.shape[0]//2:,:], axis=0)
    midpoint = np.int(histogram.shape[0]/2)
    leftx_current = 0
    rightx_current = 1040
    
    #cv2.imshow("img", img)
    #cv2.imshow("lane", lane)
    #print(histogram)
    chk_arry = []
    cnt = 0
    while cnt < len(histogram) - 1 and len(chk_arry) < 2:
        arry = []
        t = 0
        for j in range(cnt, histogram.shape[0]):
            if histogram[j] >= 1000:
                if j == (len(histogram) - 1):
                    cnt = j
                    break
                arry.append(j)
                t = 1
            elif t == 1:
                cnt = j
                break
            else:
                cnt = j
            
        if len(arry) >= 20:
            if len(chk_arry) != 0:
                is_closer = chk_arry.pop()
                if abs(is_closer[-1] - arry[0]) >= 200:
                    print('a')
                    chk_arry.append(is_closer)
                    chk_arry.append(arry)
                else:
                    print('b')
                    if is_closer[0] > arry[0]:
                        print('b1')
                        chk_arry.append(is_closer)
                    else:
                        print('b2')
                        chk_arry.append(arry)
                    
            else:
                chk_arry.append(arry)
    
    p_l = 0
    p_r = 0
    p_m = 520
    
    #print(len(chk_arry))
    #print((chk_arry))
    if len(chk_arry) == 2:
        leftx_current = np.argmax(histogram[:chk_arry[0][-1]])
        rightx_current = np.argmax(histogram[chk_arry[1][0]:]) + np.int(chk_arry[1][0])
        p_l = leftx_current
        p_r = rightx_current
        p_m = (p_l + p_r) // 2
    elif len(chk_arry) == 1:
        for chk in chk_arry:
            if len(chk_arry) == 1 and prev_l > prev_r:
                leftx_current = np.argmax(histogram[:chk[-1]])
                rightx_current = -1
                p_l = leftx_current
                p_r = prev_r_pos
                p_m = (p_l + p_r) // 2
            elif len(chk_arry) == 1 and prev_r > prev_l:
                leftx_current = -1
                rightx_current = np.argmax(histogram[chk[0]:]) + np.int(chk[0])
                p_l = prev_l_pos
                p_r = rightx_current
                p_m = (p_l + p_r) // 2
            else:
                if (chk[0]+chk[-1])//2 > midpoint:
                    leftx_current = -1
                    rightx_current = np.argmax(histogram[chk[0]:]) + np.int(chk[0])
                    p_l = prev_l_pos
                    p_r = rightx_current
                    p_m = (p_l + p_r) // 2
                elif (chk[0]+chk[-1])//2 < midpoint:
                    leftx_current = np.argmax(histogram[:chk[-1]])
                    rightx_current = -1
                    p_l = leftx_current
                    p_r = prev_r_pos
                    p_m = (p_l + p_r) // 2
                else:
                    leftx_current = np.argmax(histogram[:midpoint])
                    rightx_current = np.argmax(histogram[midpoint:]) + midpoint
                    p_l = leftx_current
                    p_r = rightx_current
                    p_m = (p_l + p_r) // 2
    else:
        leftx_current = prev_l_pos
        rightx_current = prev_r_pos
        p_l = leftx_current
        p_r = rightx_current
        p_m = (p_l + p_r) // 2
                    
    window_height = np.int(lane.shape[0]/nwindows)
    nz = lane.nonzero()

    left_lane_inds = []
    right_lane_inds = []
    num_l = 0
    num_r = 0
    
    lx, ly, rx, ry = [], [], [], []

    out_img = np.dstack((lane, lane, lane))*255

    for window in range(nwindows):
        if leftx_current < 0:
            win_yl = lane.shape[0] - (window+1)*window_height
            win_yh = lane.shape[0] - window*window_height
            
            win_xll = p_l - margin
            win_xlh = p_l + margin
            win_xrl = rightx_current - margin
            win_xrh = rightx_current + margin

            win_cl = rightx_current - 260 
            win_ch = rightx_current - 260 
            
            cv2.rectangle(out_img,((win_xll + win_xrl)//2 - 260,win_yl),((win_xll + win_xrl)//2 - 180,win_yh),(0,255,0), 2)   # left
            cv2.rectangle(out_img,(win_xrl,win_yl),(win_xrh,win_yh),(0,255,0), 2)   # right
            cv2.rectangle(out_img,(win_cl,win_yl),(win_ch,win_yh),(0,255,0), 2)     # center
            
            # cal right
            good_right_inds = ((nz[0] >= win_yl)&(nz[0] < win_yh)&(nz[1] >= win_xrl)&(nz[1] < win_xrh)).nonzero()[0]
            right_lane_inds.append(good_right_inds)
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nz[1][good_right_inds]))
                num_r += 1
            lx.append((win_xll + win_xrl)//2 - 220 -200)
            ly.append((win_yl + win_yh)/2)
            rx.append(rightx_current -200)
            ry.append((win_yl + win_yh)/2)
            
        elif rightx_current < 0:
            win_yl = lane.shape[0] - (window+1)*window_height
            win_yh = lane.shape[0] - window*window_height

            win_xll = leftx_current - margin
            win_xlh = leftx_current + margin
            win_xrl = p_r - margin
            win_xrh = p_r + margin
            
            win_cl = leftx_current + 260
            win_ch = leftx_current + 260
            
            cv2.rectangle(out_img,(win_xll,win_yl),(win_xlh,win_yh),(0,255,0), 2)   # left
            cv2.rectangle(out_img,((win_xll + win_xrl)//2 + 260,win_yl),((win_xll + win_xrl)//2 + 340,win_yh),(0,255,0), 2)  # right
            cv2.rectangle(out_img,(win_cl,win_yl),(win_ch,win_yh),(0,255,0), 2)     # center
            
            # cal left
            good_left_inds = ((nz[0] >= win_yl)&(nz[0] < win_yh)&(nz[1] >= win_xll)&(nz[1] < win_xlh)).nonzero()[0]
            left_lane_inds.append(good_left_inds)
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nz[1][good_left_inds]))
                num_l += 1
            lx.append(leftx_current-200)
            ly.append((win_yl + win_yh)/2)
            rx.append((win_xll + win_xrl)//2 + 300 -200)
            ry.append((win_yl + win_yh)/2)
            
        else:
            win_yl = lane.shape[0] - (window+1)*window_height
            win_yh = lane.shape[0] - window*window_height

            win_xll = leftx_current - margin
            win_xlh = leftx_current + margin
            win_xrl = rightx_current - margin
            win_xrh = rightx_current + margin

            win_cl = (leftx_current + rightx_current) // 2
            win_ch = (leftx_current + rightx_current) // 2

            #draw windows
            cv2.rectangle(out_img,(win_cl,win_yl),(win_ch,win_yh),(0,255,0), 2)     # center
            
            # cal left
            good_left_inds = ((nz[0] >= win_yl)&(nz[0] < win_yh)&(nz[1] >= win_xll)&(nz[1] < win_xlh)).nonzero()[0]
            left_lane_inds.append(good_left_inds)
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nz[1][good_left_inds]))
                num_l += 1
                cv2.rectangle(out_img,(win_xll,win_yl),(win_xlh,win_yh),(0,255,0), 2)   # left
            lx.append(leftx_current-200)
            ly.append((win_yl + win_yh)/2)
            
            # cal right
            good_right_inds = ((nz[0] >= win_yl)&(nz[0] < win_yh)&(nz[1] >= win_xrl)&(nz[1] < win_xrh)).nonzero()[0]
            right_lane_inds.append(good_right_inds)
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nz[1][good_right_inds]))
                num_r += 1
                cv2.rectangle(out_img,(win_xrl,win_yl),(win_xrh,win_yh),(0,255,0), 2)   # right
            rx.append(rightx_current-200)
            ry.append((win_yl + win_yh)/2)
    #print(prev_l, prev_r, num_l, num_r)
    prev_l = num_l
    prev_r = num_r
    prev_l_pos = p_l
    prev_r_pos = p_r
    prev_mid_pos = p_m
    
    
    if num_l == 0:
        right_lane_inds = np.concatenate(right_lane_inds)
        lx = [0, 0]
        ly = [480, 479]
        lfit = np.polyfit(np.array(ly),np.array(lx),2)
        rfit = np.polyfit(np.array(ry),np.array(rx),2)
        out_img[nz[0][right_lane_inds] , nz[1][right_lane_inds]] = [0, 0, 255]
        cv2.imshow("viewer", out_img)
    elif num_r == 0:
        left_lane_inds = np.concatenate(left_lane_inds)
        rx = [640, 640]
        ry = [480, 479]
        lfit = np.polyfit(np.array(ly),np.array(lx),2)
        rfit = np.polyfit(np.array(ry),np.array(rx),2)
        out_img[nz[0][left_lane_inds], nz[1][left_lane_inds]] = [255, 0, 0]
        cv2.imshow("viewer", out_img)
    else:
        #total of search data
        left_lane_inds = np.concatenate(left_lane_inds)
        lfit = np.polyfit(np.array(ly),np.array(lx),2)
        right_lane_inds = np.concatenate(right_lane_inds)
        rfit = np.polyfit(np.array(ry),np.array(rx),2)
        #left_fit = np.polyfit(nz[0][left_lane_inds], nz[1][left_lane_inds], 2)
        #right_fit = np.polyfit(nz[0][right_lane_inds] , nz[1][right_lane_inds], 2)
        out_img[nz[0][left_lane_inds], nz[1][left_lane_inds]] = [255, 0, 0]
        out_img[nz[0][right_lane_inds] , nz[1][right_lane_inds]] = [0, 0, 255]
        cv2.imshow("viewer", out_img)
    
    return lfit, rfit, num_l, num_r

def draw_lane(image, warp_img, Minv, left_fit, right_fit):
    warp_img = warp_img.astype(np.uint8)
    global Width, Height
    global prev_l, prev_r, prev_l_pos, prev_r_pos
    yMax = warp_img.shape[0]
    ploty = np.linspace(0, yMax - 1, yMax)
    color_warp = np.zeros_like(warp_img).astype(np.uint8)
    temp = np.zeros((480, 640, 3), np.uint8)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))]) 
    pts = np.hstack((pts_left, pts_right))
    
    color_warp = cv2.fillPoly(temp, np.int_([pts]), (0, 255, 0))
    newwarp = cv2.warpPerspective(color_warp, Minv, (Width, Height))
    
    cv2.line(color_warp, (int(left_fitx[0]), 0), (int(left_fitx[-1]), 480), (0,0,255), 1)
    cv2.line(color_warp, (int(left_fitx[0]), 0), (int(left_fitx[0]), 480), (0,0,255), 1)
    cv2.line(color_warp, (int(right_fitx[0]), 0), (int(right_fitx[-1]), 480), (0,0,255), 1)
    cv2.line(color_warp, (int(right_fitx[0]), 0), (int(right_fitx[0]), 480), (0,0,255), 1)
    l_sum = 0
    r_sum = 0
    for idx, i in enumerate(reversed(pts_left[0])):
        if idx == 4:
            break
        l_sum += i
    for idx, i in enumerate(pts_right[0]):
        if idx == 4:
            break
        r_sum += i
    print(l_sum[0] // 4, r_sum[0] // 4)
    #print(pts_left[0][-1], pts_right[0][0])
    #cv2.imshow("color_warp", color_warp)
    #cv2.imshow("newwarp", newwarp)
    #print(left_fit)
    return cv2.addWeighted(image, 1, newwarp, 0.3, 0), l_sum[0].astype(int) / 4, r_sum[0].astype(int) / 4

# record videos
# fourcc = cv2.VideoWriter_fourcc(*'DIVX')
# out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (int(Width), int(Height)))

def start():
    global Width, Height, cap, prev_l, prev_r

    _, frame = cap.read()
    while not frame.size == (Width*Height*3):
        _, frame = cap.read()
        continue

    print("start")
    fps = 0
    total_fps = 0
    while cap.isOpened():
        
        _, frame = cap.read()
        fps += 1
        total_fps += 1
        image = calibrate_image(frame)
        warp_img, M, Minv = warp_image(image, warp_src, warp_dist, (warp_img_w, warp_img_h))
        left_fit, right_fit, cnt_left, cnt_right = warp_process_image(warp_img)
        lane_img, result_x, result_y = draw_lane(image, warp_img, Minv, left_fit, right_fit)
        print(prev_l, prev_r)
        if fps == 30:
            if result_x == -1:
                result_x = 0
            elif result_y == -1:
                result_y = 0
            f.write(str(result_x) + ", " + str(result_y) + "\n")
            fps = 0
        cv2. putText(lane_img, str(total_fps), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
        cv2.imshow(window_title, lane_img)
        #out.write(lane_img)
        if total_fps == 3121:
            cv2.waitKey()
        else:
            cv2.waitKey(1)
        #exit()
        

if __name__ == '__main__':
    start()
    cap.release()
    #out.release()
    cv2.destroyAllWindows()