
import image_annotation as cva
import cv2 as cv          #OpenCV 
import numpy as np        #Numpy for math operations and arrays
import matplotlib.pyplot as plt 
import skimage 
import scipy.interpolate as si
import sknw 
import networkx as nx
import scipy.ndimage as ndi
import time
#import pylab as pl



#moving average
def running_mean_uniform_filter1d(x, N):
    x = np.pad(x, (N//2, N-1-N//2), mode='edge') #pad so we get the same array length back
    return ndi.uniform_filter1d(x, N, mode='nearest', origin=-(N//2))[:-(N-1)] #some fancy filtering for moving mean

### HELPER FUNCTIONS FOR DEBUGGING

#dummy for GUI
def callback(x): 
    return

#function to view img, debug only
def tv(img, time=0, close=True, name="img", size = 1): 
    cv.namedWindow(name, cv.WINDOW_GUI_EXPANDED)
    cv.resizeWindow(name, int(img.shape[1]*size), int(img.shape[0]*size))
    cv.imshow(name,img)
    cv.waitKey(time)
    if close == True:
        cv.destroyWindow(name)
    return

### FUNCTIONS NEEDED FOR PROGRAM

#sorts lines ascending, according to distance of closest endpoint to origin in X-direction, sorts points so that first point is always top point            
def sortLinesLeftToRightTopPointFirst(lines): 
    left_x = []
    for i in range(len(lines)): #go through the lines and their points, 
        l = lines[i][0] #pick the vector that represents (x0,y0,x1,y1), for some reason lines structure is [[[x0.0 y0.0 x1.0 y1.0]][[x0.1 y0.1 x1.1 y1.1]]] 
        if l[0] < l[2]: # append the lower x_value of the two points to our list , this is the most left x of our two points
            left_x.append(l[0])
        else:
            left_x.append(l[2])

        if l[1] > l[3]: #if the second point is higher than the first point, swap their positions, so we get the higher point first
            lines[i][0] = [l[2], l[3], l[0], l[1]] 
        
    left_x = np.array(left_x) #turn it into numpy array
    #print("left_x:")
    #print(left_x)
    indx = left_x.argsort() #get an array with the sorted index of distance array
    #print("indx:")
    #print(indx)
    lines = lines[indx] #sort the lines with ascending distance of the left point to the origin in x direciton 
    return lines

### Functions returns point coordinates of last point at top of image from list of edge coordinates
# CAUTION - Only works for cooridnates list that are aquired through np.nonzero(left_contour_img) / returns values in row-major C-style order
def getLastPointBottom(coordinates):
    point = (0,0)
    l = len(coordinates[0])-1 #helper to go through list backwards
    if coordinates[0][l] == coordinates[0][l-1]: #check if we have multiple pixels on lowest y-row
        #print("getLastPointBottom, found multiple pixels in last row")
        r_x_last_row = coordinates[1][l] # the x-coordinate of the last, thus most right pixel we found
        for i in range(l,0,-1): #count back wards
            last_idx_same_row = i
            if coordinates[0][i] != coordinates[0][i-1]: #if next value has another y-value -> is in next row, we found last entry for last row
                break #end for loop 
        #Now we can compare x-values to see if line moves left or right and 
        # which pixel is endpoint of our line 
        x_next_row = coordinates[1][last_idx_same_row-1] 
        if r_x_last_row < x_next_row: #we move to the left, so first entry for last row is end point 
            point = (coordinates[1][last_idx_same_row],coordinates[0][last_idx_same_row])
        else: #we move to right, last entry is actual end point
            point = (coordinates[1][-1],coordinates[0][-1])


    else: #if we don't have multiple pixels, last entry is actual end point
        #print("getLastPointBottom, only one pixel in last row")
        point = (coordinates[1][-1],coordinates[0][-1])
        
    return point #point (x,y)

### Functions returns point coordinates of last point at top of image from list of edge coordinates
def getLastPointTop(coordinates):
    point = (0,0)
    l = len(coordinates[0])-1 #helper to go through list backwards
    if coordinates[0][0] == coordinates[0][1]: #check if we have multiple pixels on top y-row
        l_x_last_row = coordinates[1][0] # the x-coordinate of the first, thus most left pixel we found
        for i in range(0,l,1): #count through, forward
            last_idx_same_row = i
            if coordinates[0][i] != coordinates[0][i+1]: #if next value has another y-value -> is in next row, we found last entry for last row
                last_idx_same_row = i
                break #end for loop 
        #Now we can compare x-values to see if line moves left or right and 
        # which pixel is endpoint of our line 
        x_next_row = coordinates[1][last_idx_same_row+1] 
        if l_x_last_row > x_next_row: #we move to the right, so first entry for last row is end point
            point = (coordinates[1][last_idx_same_row],coordinates[0][last_idx_same_row])
        else: #we move to the left, thus first entry in array is actually end point of edge
            point = (coordinates[1][0],coordinates[0][0])
            
    else: #we dont have multiple pixels or we move to the left, thus first entry in array is actually end point of edge
        point = (coordinates[1][0],coordinates[0][0])
    
    return point #point (x,y)

### Function calculates point A of a isosceles triangles with points B,C and angle a 
def getTrianglePoint(B, C, a):
    #A,B,C form isosceles triangles, B,C are known points
    # Point A will be calculated, and is left to BC line 
    #     A  , a is angle in this corner, can be selected 
    #   .  .
    #  .    .
    # B------C  

    #first calc distance between A and B 
    distBC = np.sqrt(np.square(B[0]-C[0])+np.square(B[1]-C[1]))
    #Calculate height of triangle for given angle a 
    h = ( (distBC/2) / (np.tan((a/2)*np.pi/180)) ) #np*pi/180 to go from degres to radians for np.tan function 
    An = ((distBC/2),-h)    # Ax = Bx + distBC/2 ; Ay = By - h , minus h because of opencv coordinate directions 
    #print("An:", An) #debug only
    #Now rotate with respect to angle on BC line to X-Axis in original coordinate system
    angle = np.arctan2((C[1]-B[1]),(C[0]-B[0])) # angle in radians, we can keep it like that as we feed it back to np. functions
    #print("angle:", (angle*180/np.pi)) #debug only
    Anrx = An[0]*np.cos(angle) - An[1]*np.sin(angle) # x' = x cos(a) - y sin(a) 
    Anry = An[0]*np.sin(angle) + An[1]*np.cos(angle)# y' = x sin(a) - y cos(a)
    #print("Anrx:", Anrx) #debug only
    #print("Anry:", Anry) #debug only
    #Finaly translate into original coordinate system
    A = ((int(Anrx + B[0])), (int(Anry + B[1]))) #our point in original coordinate system , convert to int so we get discrete pixel
    # not perfect, some rounding errors by int(), but ok for our purpose 
    return A 

#for contour end line cut only 
def getlinetp(x1, y1, x2, y2):
    #check if we have vertikal line
    if x1 == x2:
        m = np.inf # slope of line between given points
        b = np.nan # offset for line between given points
        mn = 0 # slope of normal line to line between given points
        bn = y1 # and its offset 
    #check if we have horizontal line
    elif y1 == y2:
        m = 0
        b = y1
        mn = np.inf
        bn = np.nan
    #if not vertical nor horizontal
    else:
        m = (y2-y1)/(x2-x1)
        b = y1-m*x1
        mn = -1/m
        bn = y1-mn*x1 
    
    return m, b, mn, bn

#use linear regression, gives better solution than two point approach 
def getlinereg(x, y): # x = [x1, x2...], y = [y1, y2...]
    #check if we have vertical line
    if np.all(x==x[0]):
        m = np.inf # slope of line between given points
        b = np.nan # offset for line between given points
        mn = 0 # slope of normal line to line between given points
        bn = y[0] # and its offset 
    #check if we have horizontal line
    elif np.all(y==y[0]):
        m = 0
        b = x[0]
        mn = np.inf
        bn = np.nan
    #if not vertical nor horizontal
    else:
        # proper solution for the least-square fit of y = a*x is just a=x.dot(y)/x.dot(x) 
        xd = x - x[0] #have to offset, otherwise get wrong result
        yd = y - y[0]
        m = xd.dot(yd)/xd.dot(xd)
        b = y[0]-m*x[0]
        mn = -1/m
        bn = y[0]-mn*x[0]
    
    return m, b, mn, bn

#same like getlinereg but with intersect point on index i
def getlineregI(x, y, i): # x = [x1, x2...], y = [y1, y2...]
    #check if we have vertical line
    if np.all(x==x[i]):
        m = np.inf # slope of line between given points
        b = np.nan # offset for line between given points
        mn = 0 # slope of normal line to line between given points
        bn = y[i] # and its offset 
    #check if we have horizontal line
    elif np.all(y==y[i]):
        m = 0
        b = x[i]
        mn = np.inf
        bn = np.nan
    #if not vertical nor horizontal
    else:
        # proper solution for the least-square fit of y = a*x is just a=x.dot(y)/x.dot(x) 
        xd = x - x[i] #have to offset, otherwise get wrong result
        if abs(max(xd)-min(xd)) <= 2: #sometimes we are almost vertical but not x==x[i] => leads to jumpy, inaccurate result -> check we have reasonable spread in xd
            m = np.inf # treat like  vertical line case
            b = np.nan 
            mn = 0 
            bn = y[i] 
        else:
            yd = y - y[i]
            m = xd.dot(yd)/xd.dot(xd)
            b = y[i]-m*x[i]
            mn = -1/m
            bn = y[i]-mn*x[i]

    return m, b, mn, bn

# from: https://stackoverflow.com/questions/34803197/fast-b-spline-algorithm-with-numpy-scipy
def scipy_bspline(ar, n=100, degree=3, periodic=False):
    """ Calculate n samples on a bspline

        ar :      Array ov control vertices
        shape ar = np.array([[ 50.,  25.],
                [ 59.,  12.],
                [ 50.,  10.],
                [ 57.,   2.],
                [ 40.,   4.],
                [ 40.,   14.]])
        n  :      Number of samples to return
        degree:   Curve degree
        periodic: True - Curve is closed
    """
    ar = np.asarray(ar)
    count = ar.shape[0]

    # Closed curve
    if periodic:
        kv = np.arange(-degree,count+degree+1)
        factor, fraction = divmod(count+degree+1, count)
        ar = np.roll(np.concatenate((ar,) * factor + (ar[:fraction],)),-1,axis=0)
        degree = np.clip(degree,1,degree)

    # Opened curve
    else:
        degree = np.clip(degree,1,count-1)
        kv = np.clip(np.arange(count+degree+1)-degree,0,count-degree)

    # Return samples
    max_param = count - (degree * (1-periodic))
    spl = si.BSpline(kv, ar, degree)
    return spl(np.linspace(0,max_param,n))

#taken from https://stackoverflow.com/questions/22029548/is-it-possible-in-opencv-to-plot-local-curvature-as-a-heat-map-representing-an-o and heavily modified
def compute_pointness(I, C):
    
    # Compute gradients
    # GX = cv2.Sobel(I, cv2.CV_32F, 1, 0, ksize=5, scale=1)
    # GY = cv2.Sobel(I, cv2.CV_32F, 0, 1, ksize=5, scale=1)
    GX = cv.Scharr(I, cv.CV_32F, 1, 0, scale=1)
    GY = cv.Scharr(I, cv.CV_32F, 0, 1, scale=1)
    GX = GX + 0.0001  # Avoid div by zero


    #magnitude = np.sqrt((GX ** 2) + (GY ** 2))
    #calculat the orientation of the gradient
    orientation = np.arctan2(GY, GX) * (180 / np.pi) % 180


    # pl.figure()
    # pl.imshow(GX, cmap=pl.cm.jet)
    # pl.figure()
    # pl.imshow(GY, cmap=pl.cm.jet)
    # pl.figure()
    # pl.imshow(magnitude, cmap=pl.cm.jet)
    # pl.figure()
    # pl.imshow(orientation, cmap=pl.cm.jet)
    
    

    #old, from stackoverflow
    # heatmap = np.zeros_like(I, dtype=float)
    # pointed_points = []
    # measure = []
    # N = len(C)
    # for i in range(N):
    #     x1, y1 = C[i,0,0] , C[i,0,1]
    #     x2, y2 = C[((i + n)% N,0,0)], C[((i + n) % N,0,1)]

    #     # Angle between gradient vectors (gx1, gy1) and (gx2, gy2)
    #     gx1 = GX[y1, x1]
    #     gy1 = GY[y1, x1]
    #     gx2 = GX[y2, x2]
    #     gy2 = GY[y2, x2]
    #     cos_angle = gx1 * gx2 + gy1 * gy2
    #     cos_angle /= (np.linalg.norm((gx1, gy1)) * np.linalg.norm((gx2, gy2)))
    #     angle = np.arccos(cos_angle)
    #     if cos_angle < 0:
    #         angle = np.pi - angle

        
    #     x1, y1 = C[((2*i + n) // 2) % N, 0, 0], C[((2*i + n) // 2) % N, 0, 1]  # Get the middle point between i and (i + n)
    #     heatmap[y1, x1] = angle  # Use angle between gradient vectors as score
    #     measure.append((angle, x1, y1, gx1, gy1))
    N = len(C)
    out = np.zeros(shape=(N,4))
    heatmap2 = np.zeros_like(I, dtype=float)
    for i in range(N):
        x1, y1 = C[i,0,0] , C[i,0,1] #get x, y coordiantes of contour
        ori = orientation[y1, x1] # get the orientation of the gradient at that position
        gy = GY[y1, x1] # get the value of the y gradient at that position
        heatmap2[y1, x1] = ori  #for visualisation
        out[i] = [ori, gy, x1, y1]
        #out.append([ori, gy, x1, y1]) #save the orientation

    #_, x1, y1, gx1, gy1 = max(measure2)  # Most pointed point for each contour, old stackoverflow
    #_, x1, y1 = min(measureOut) #find minimum of gradient orientation along our contour => that is where our reflection or already deposited structure is
    # as the vector of the orientation changes from pointing up to pointing down (or other way around, what matters we go through 0 degrees)
    
    # old, Possible to filter for those blobs with measure > val in heatmap instead.
    #old, pointed_points.append((x1, y1, gx1, gy1))
   # pointed_points.append((x1, y1, 0, 0)) #save our position, 
 
    heatmap = cv.GaussianBlur(heatmap2, (3, 3), heatmap2.max()) #visualisation stuff
    return heatmap, out

#taken from https://stackoverflow.com/questions/22029548/is-it-possible-in-opencv-to-plot-local-curvature-as-a-heat-map-representing-an-o for compute_pointness function visualisation
def plot_points(image, pointed_points, radius=2, color=(255, 0, 0)):
    for [_, _, x1, y1,] in pointed_points:
        cv.circle(image, (x1, y1), radius, color, -1)


#get jet contours, edges
def getJetContoursGradient(image, collector, nozzleleftlowy, nozzlerightlowy, nozzleleftlowx, nozzlerightlowx, Threshold_kernel=101, Blur_kernel=3, upper_cut_off=0, lower_cut_off=1000):
    try:
        
        #optionaly implement gaussian blur before edge detection to reduce noise if needed
        if Blur_kernel != 0:
            image = cv.GaussianBlur(image,(Blur_kernel,Blur_kernel),0)


        img_width = image.shape[1]
        img_height = image.shape[0]
        imgedge_pixel_dist = 10
        #image = cv.bitwise_not(image) #test for cheap cam
        #21 kernel for JC1 sometimes ok, 101 seems saver overall
        adgimage = cv.adaptiveThreshold(image,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,Threshold_kernel,2) #101 kernel seems to work fine with low contrast images #better than mean
        #tv(adgimage, size=0.5)
        adgimage[:upper_cut_off,:] = 0 #set everything above upper cut off black -> so we won't find any contours there
        adgimage[:,:imgedge_pixel_dist] = 0 #set 10 pixel from left edge black, so we dont run into image edge
        adgimage[:,img_width-imgedge_pixel_dist:] = 0 #set 10 pixel from right edge black, so we dont run into image edge
        kernel = np.ones((3,3),np.uint8)
        adgimage = cv.morphologyEx(adgimage, cv.MORPH_OPEN, kernel, iterations=2) #remove some of the noise specs 

        #tv(adgimage, size=0.5)
        #adgc = cv.cvtColor(adgimage, cv.COLOR_GRAY2BGR) #for debug only
        
        #fill image black below collector line to get rid of reflections
        adgimage[collector:,:] = 0
        #tv(adgimage, size=0.5)

        #horizontal line should use other reference, for now
        collector2 = np.rint(collector/2)
        

        #tv(adgimage, size=0.5)
        #some variables for hough line detection, selected by trial and error 
        HLrho=1
        HLtheta=(np.pi/180)
        HLthreshold=50
        HLminLineLength=100 #was 100
        HLmaxLineGap=100 #was 500
        HorizontalAngleThres=3 # throw away lines that are not almost horizontal
        lines = cv.HoughLinesP(adgimage, rho=HLrho, theta=HLtheta, threshold=HLthreshold, minLineLength=HLminLineLength, maxLineGap=HLmaxLineGap) 
        if lines is not None: 
            horizontal_lines = [] #create a list for horizontal lines
            for i in range(len(lines)): #go through the list of lines
                l = lines[i][0] #pick the vector that represents (x0,y0,x1,y1), for some reason lines structure is [[[x0.0 y0.0 x1.0 y1.0]][[x0.1 y0.1 x1.1 y1.1]]] .0 first line, .1 second line...
                
                #cv.line(adgc, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv.LINE_AA) #draw all lines, for debugging
                #cv.drawMarker(adgc, (l[0], l[1]), (255,0,0), markerType = cv.MARKER_SQUARE, markerSize= 20) #should be top, debug only
                #cv.drawMarker(adgc, (l[2], l[3]), (0,255,0), markerType = cv.MARKER_TILTED_CROSS, markerSize= 20) #should be bottom, debug only
                #tv(adgc, size=0.5)

                if abs(np.arctan2((l[3]-l[1]),(l[2]-l[0])) *180 /np.pi  ) < HorizontalAngleThres and l[1] > collector2 and l[3] > collector2: #if the angle of the line is greater horizontal_angle_thres, -> look for (almost) horizontal lines
                    if l[0] > l[1]: #if the first point further left than the second point, swap them. Sorts the line points so that first point is always on the left 
                        lines[i][0] = [l[2], l[3], l[0], l[1]] 

                    horizontal_lines.append(l) #append line to horizontal line list 
                    #print(np.arctan2((l[3]-l[1]),(l[2]-l[0])) *180 /np.pi) #debug only
                    #cv.line(timg, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv.LINE_AA) #draw the vertical line, debugging only
                    #tv(timg) #debuggin only
            
            #tv(adgc, size=0.5)
            y_min = img_height #init with lower image edge
            if horizontal_lines is not None:
                for i in range(len(horizontal_lines)): #go through horizontal lines, and find upmost y_value in start and end points (left and right points)
                    l = horizontal_lines[i] #pick the vector that represents (x0,y0,x1,y1)
                    y_min = min(l[1],l[3], y_min)
                    #cv.line(adgc, (0, y_min), (img_width, y_min), (0,0,255), 1, cv.LINE_AA)


        adgimage = cv.morphologyEx(adgimage, cv.MORPH_CLOSE, kernel, iterations=3)

        contours, hierarchy = cv.findContours(adgimage, mode = cv.RETR_TREE, method = cv.CHAIN_APPROX_NONE) #joining continuous points having same color/intesity, returns a list of the contours, represented as x/y coordinate arrays. 
        all_contours = contours
        #RETR_LIST returns all contours, method to apporxe_none to store all contour points
        #RETR_EXTERNAL, returns only outer extreme contours
        #RETR_TREE gives full hierarchy, needed so we can detect if inner or outer contour => clockwise or counterclockwise point array

        #print(contours)
        #contours = sorted(contours, key=lambda x: (cv.boundingRect(x))[1])#sort contours increasing y-value upper left point of bouning rectangle boundingRect returns: top left point x, y, width, heigth
        #New version with hiearchy:
        sort_idx = list(range(len(contours))) #create a list with 0,1,2,3..... for length of contours
        s = sorted(zip(contours,sort_idx), key=lambda x: (cv.boundingRect(x[0])[2]*cv.boundingRect(x[0])[3]), reverse=True)#sort contours decreasing y-value upper left point of bouning rectangle boundingRect returns: top left point x, y, width, heigth
        contours, sort_idx = zip(*s) #unzip contours and sort_idx, sort_idx is sorted like contours, so it gives us new index order
        contours = list(contours) #turn tuple into list type for following sorting
        sort_idx = np.array(sort_idx) #turn it into numpy array
        hierarchy[0][sort_idx] #use sort_idx to sort hierarchy in same way our contours are now sorted
        hierarchy = list(hierarchy[0]) #turn into list of arrays, so we can also sort it in the following

        remove_idx = []
        nozzle_dx_1_4 = abs(nozzleleftlowx - nozzlerightlowx)/4
        ll_limit = nozzleleftlowx - nozzle_dx_1_4   #   | |--nozzle--| |
        lr_limit = nozzleleftlowx + nozzle_dx_1_4   #   | |--nozzle--| |
        rl_limit = nozzlerightlowx - nozzle_dx_1_4  #   | |--nozzle--| |
        rr_limit = nozzlerightlowx + nozzle_dx_1_4  #   | | |      | | |
                                                    #  ll   lr     rl  rr 

        for i in range(len(contours)): #remove contours based on some criteria 
            #print(i)
            #print(cv.boundingRect(contours[i])[1])
            #boundingRect returns: top left point x, y, width, heigth
            x_up_left = cv.boundingRect(contours[i])[0]
            y_up_left = cv.boundingRect(contours[i])[1]
            box_width = cv.boundingRect(contours[i])[2]
            box_height = cv.boundingRect(contours[i])[3]
            #print("contour area:", box_width*box_height)
            #nozzleleftlowx, nozzlerightlowx
            if y_up_left + box_height < upper_cut_off: #if the lower bounding box edge is above our cut_off => remove it from list, is fragmented nozzle contour
                remove_idx.append(i) 
            elif y_up_left > lower_cut_off: # if the contours upper left point starts below the lower cut off, it is not attached to nozzle/starts below it, remove it. could e.g. be a contour inside the jet/blob
                remove_idx.append(i)
            #check if the upper left or right corner of the bounding box is within 1/4 width of nozzle from left/right lower nozzle edge, in x-direction
            # this removes contours that dont't start close to nozzle 
            elif not ( (not (ll_limit<x_up_left<lr_limit) and not(rl_limit<(x_up_left+box_width)<rr_limit )) or (not (ll_limit<x_up_left+box_width<lr_limit) and not(rl_limit<x_up_left<rr_limit ))):
                remove_idx.append(i)
            
        if len(remove_idx) != 0:
            for index in sorted(remove_idx, reverse=True):
                del contours[index]
                del hierarchy[index]
        
        #box_width = cv.boundingRect(contours[0])[2]
        #box_height = cv.boundingRect(contours[0])[3]
        #print("remaining contour area:", box_width*box_height)
        #print(type(contours)) #debug only
        #print(contours) #debug only
        #print("number of contours:", len(contours))
        #print(len(hierarchy))



        if len(contours) == 0:
            status = "no valid contour found"
            left_contour = 0
            right_contour = 0

        else: #always pick first contour, should be the biggest one (sometimes we detect almost the same contour multiple times)
        #print("handling one contour case")


            status = "Ok"
            contour_direction = hierarchy[0][3]
            #print("hieararchy", contour_direction)
            single_contour = contours[0]
            #contour direction => -1 = counter-clockwise, != -1 => clockwise
            if contour_direction != -1: #if order of coordiante points is not counter clockwise, we turn it around
                single_contour = single_contour[::-1]
                #print("flipped direction")

            #find potential deposition point / jet contour end, so we know where to cut off e.g. jet reflection or pritned structure that was included in contour
            #print("attempt to find dep point")
            #does some gradient calulations, orientation_gradY_points gives us array of [gradient vector orientation[deg], gradient in y-direction, img X position, img Y position] of each point along contour 
            heatmap, orientation_gradY_points = compute_pointness(image, contours[0])
            #now search for the entries that are below our upper horizontal line => that is where we will search 
            id_y_below = np.where(orientation_gradY_points[:,3] > y_min) 
            #the y-gradient on the jet side that is deposited must be positive, and in the reflection/attachment point we should see a change to negative (or zero?!) values
            # can be best visualised: 
            # pl.figure()
            # pl.imshow(heatmap, cmap=pl.cm.jet)
            # pl.colorbar()
            # I_color = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
            # #grady_zero_points = []
            # print(orientation_gradY_points[0][1])
            # for i in range(len(orientation_gradY_points)):
            #    # if orientation_gradY_points[i][1] == 0:
            #         #grady_zero_points.append(orientation_gradY_points[i])    # old
            #     if orientation_gradY_points[i][1] > 0: #positive gradient in y => red point
            #         cv.circle(I_color, (int(orientation_gradY_points[i][2]), int(orientation_gradY_points[i][3])), 1, (255,0,0), -1)
            #     if orientation_gradY_points[i][1] < 0: #pnegative gradient in y => blue point
            #         cv.circle(I_color, (int(orientation_gradY_points[i][2]), int(orientation_gradY_points[i][3])), 1, (0,0,255), -1)
                
            # cv.line(I_color, (0, collector), (img_width, collector), (255,0,0), 1, cv.LINE_AA) #lower bound search area (not yet implemented in logic, lets see if we have to check boundary)
            # cv.line(I_color, (0, y_min), (img_width, y_min), (255,0,0), 1, cv.LINE_AA) #upper bound search area 
            #plot_points(I_color, grady_zero_points, radius = 4, color=(0,255,0))
            # pl.figure()
            # pl.imshow(I_color)
            # pl.show()

            #cv.circle(I_color, (int(orientation_gradY_points[id_y_below[0][0]][2]), int(orientation_gradY_points[id_y_below[0][0]][3])), 8, (255,255,255), -1)
            #cv.circle(I_color, (int(orientation_gradY_points[id_y_below[0][-1]][2]), int(orientation_gradY_points[id_y_below[0][-1]][3])), 8, (255,255,0), -1)


            # so now we check on which side we have positive gradient, if it is at the end of our index list, flip the list 
            if orientation_gradY_points[id_y_below[0][-1],1] > 0:
                id_y_below = np.flip(id_y_below)

            # moving average over our gradient, so we supress single pixels with zero or negative gradient that can throw off finding the correct point
            orientation_gradY_points[:,1] = running_mean_uniform_filter1d(orientation_gradY_points[:,1], 5)
            
            # now we go along our contour (according to index list), and look for change from positive to neg
            y_max = 0
            for i in id_y_below[0]:
                if orientation_gradY_points[i,1] <= 0:
                    y_max = int(orientation_gradY_points[i,3])
                    #print(orientation_gradY_points[i,1])
                    #cv.circle(I_color, (int(orientation_gradY_points[i][2]), int(orientation_gradY_points[i][3])), 8, (0,255,0), -1)
                    break
            
            if y_max == 0: #in case we didnt find a y_max with our gradient method
                print("No gradient based y_max")
                y_max = np.max(single_contour[:,0,1]) #get max y_value
                        
            
            # cv.line(I_color, (0, y_max), (img_width, y_max), (0,255,0), 1, cv.LINE_AA) #upper bound search area 
            # pl.figure()
            # pl.imshow(I_color)
            # pl.show()

            # tv(I_color, size = 0.5)
            # tv(heatmap, size=0.5)
            #print("ymax", y_max)

            #print(single_contour)
            #print(single_contour[len(single_contour)//2])
            x_start = single_contour[0,0,0] 
            #check if our contour start point is on left or right side of nozzle, changes depending on how high the contour goes on each side (higher side is start side)
            if (ll_limit<x_start<lr_limit): 
                start_side = 0
                #print("start left")
                #get list of indices where our x value is in the range accepted around the right nozzle edge
                id_other_side_x = np.where(np.logical_and(single_contour[:,0,0]>rl_limit, single_contour[:,0,0]<rr_limit))[0]
                #search in that indices range for the lowest y_value => highest point
                y_max_other_side = np.where(single_contour[id_other_side_x[0]:id_other_side_x[-1],0,1] == np.min(single_contour[id_other_side_x[0]:id_other_side_x[-1],0,1]) )[0][0]
                #calculate the index of that point, by adding the offset created by the list 
                id_top_other_side = id_other_side_x[0]+y_max_other_side
            else:
                start_side = 1
            # print("start right")
                #since we are on right side, we have to flip it around to go clock wise, thus along lower side of contour 
                single_contour = single_contour[::-1] 
                #get list of indices where our x value is in the range accepted around the left nozzle edge
                id_other_side_x = np.where(np.logical_and(single_contour[:,0,0]>ll_limit, single_contour[:,0,0]<lr_limit))[0]
                #search in that indices range for the lowest y_value => highest point
                y_max_other_side = np.where(single_contour[id_other_side_x[0]:id_other_side_x[-1],0,1] == np.min(single_contour[id_other_side_x[0]:id_other_side_x[-1],0,1]) )[0][0]
                #calculate the index of that point, by adding the offset created by the list 
                id_top_other_side = id_other_side_x[0]+y_max_other_side
                    
            single_contour = single_contour[:id_top_other_side] #take only part of contour until top of other side, after that the contour wraps around/runs back on inside of edge, we dont need that part
            
  
            idx_max_y = np.where(single_contour[:,0,1] == y_max)[0] #get list of index where our y is equal y_max

            #cv.line(adgc, (0, y_max), (img_width, y_max), (255,0,0), 1, cv.LINE_AA) #debug only
            #tv(adgc, size = 0.5)
            
            #print("idx_max_y", idx_max_y)
            if len(idx_max_y) == 1:
                idx_max_y_left = idx_max_y[0]
                idx_max_y_right = idx_max_y[0]
            else:
                if start_side == 0: # in this case we walk around counter clock wise, so first entry we find in list belongs to left side
                    idx_max_y_left = idx_max_y[0]
                    idx_max_y_right = idx_max_y[-1]
                else: # in this case we walk around clock wise, so first entry we find in list belongs to right side
                    idx_max_y_left = idx_max_y[-1]
                    idx_max_y_right = idx_max_y[0]
            
            #old, conflicts with new gradient method
            # #check if we run into image edge, if so, don't split contour according to y_max, but along edge
            # idx_imgedge_x = np.where(single_contour[:,0,0] == imgedge_pixel_dist)[0] #get list of index where our x is equal 0, left image edge
            # if len(idx_imgedge_x) == 0: 
            #     idx_imgedge_x = np.where(single_contour[:,0,0] == img_width-imgedge_pixel_dist)[0] #get list of index where our x is equal img_width, right image edge

            # #print(idx_imgedge_x)
            # k = len(idx_imgedge_x)
            # if k != 0:
            #     print("image edge case")
            #     if start_side == 0:
            #         idx_max_y_left = idx_imgedge_x[0]
            #         idx_max_y_right = idx_imgedge_x[-1]
            #     else: # in this case we walk around clock wise, so first entry we find in list belongs to right side
            #         idx_max_y_left = idx_imgedge_x[-1]
            #         idx_max_y_right = idx_imgedge_x[0]

            #old
            #idx_max_y_mid = idx_max_y[len(idx_max_y)//2] #get index at middle of list , e.g. we could have more than one pixel at y_max, we want to break apart at middle of it
            #print("mid index", idx_max_y_mid, "of", len(idx_max_y))
            
            #if we start on left side, our coordinates are sorted counter clockwise
            if start_side == 0:
                left_contour = single_contour[:idx_max_y_left] #create left part of contour 
                right_contour = single_contour[idx_max_y_right:][::-1] #create right part of contour, flip its direction to have the normal start at nozzle orientation
            #if we start on right side, our coordinates are sorted clockwise
            else:
                left_contour = single_contour[idx_max_y_left:][::-1] #create left part of contour 
                right_contour = single_contour[:idx_max_y_right] #create right part of contour, flip its direction to have the normal start at nozzle orientation
                
        
            #Calculate m and b for lower nozzle edge  #points are [[xhigh,xlow],[yhigh,ylow]] 
            if (nozzlerightlowx-nozzleleftlowx) != 0:  
                m = (nozzlerightlowy-nozzleleftlowy)/(nozzlerightlowx-nozzleleftlowy) # m = (y1-y0)/(x1-x0)
                b = nozzlerightlowy-m*nozzlerightlowx #b = yi-m*xi

            #we just plug in the edge points into our line definition and check if it is solution, if so we found the intersection point
            u_idx_left = 0
            for c in left_contour[:,0]:
                if c[1] == round(m*c[0] +b): #round() slighly faster than int(np.rint)
                    break 
                u_idx_left  += 1 

            u_idx_right = 0
            for c in right_contour[:,0]:
                if c[1] == round(m*c[0] +b):
                    break 
                u_idx_right  += 1
            
            left_contour = left_contour[u_idx_left:]
            right_contour = right_contour[u_idx_right:]

            
            #Now lets try to cut the end of the jet in a perpendicular manner, 
            # get the normal for left and right contour.
            #approx short end section of contour with a line.

            #first approx the left/right contour with a bspline to make it a bit smoother
            #number of points to create out of bspline, this will be the moothed contour
            left_bspline_points = int(len(left_contour[:,0,1])/10) #1/10 of original number of points seems to work fine
            right_bspline_points = int(len(right_contour[:,0,1])/10)
            #array shaping should be done better
            left_contour_l = (left_contour[:,0,1], left_contour[:,0,0]) #(y, x)
            o = []
            for i in range(len(left_contour_l[0])):
                c = [left_contour_l[1][i],left_contour_l[0][i]]
                o.append(c)
            
            left_contour_smoothed = scipy_bspline(o,left_bspline_points,degree=5) #degree 5 seems to work well
            x_smoothed_left = left_contour_smoothed[:,0].astype('int')
            y_smoothed_left = left_contour_smoothed[:,1].astype('int')

            right_contour_l = (right_contour[:,0,1], right_contour[:,0,0]) #(y, x)
            o = []
            for i in range(len(right_contour_l[0])):
                c = [right_contour_l[1][i],right_contour_l[0][i]]
                o.append(c)

            right_contour_smoothed = scipy_bspline(o,right_bspline_points,degree=5)
            x_smoothed_right = right_contour_smoothed[:,0].astype('int')
            y_smoothed_right = right_contour_smoothed[:,1].astype('int')

            #lower end points of our smoothed contours, should be equal the original contour (bspline runs through end points)
            x1_left = x_smoothed_left[-1] #end point x, left contour
            y1_left = y_smoothed_left[-1] #end point y, right contour
            x1_right = x_smoothed_right[-1] #end point x, right contour
            y1_right = y_smoothed_right[-1] #end point y, right contour

            # how many of our smoothed points to use to approximate slope of end section
            approx_dist = 5 #five seems to work fine 
            #get the slope and offset of the end section and of the normal 
            ml, bl, mnl, bnl = getlinereg(x_smoothed_left[-approx_dist:][::-1], y_smoothed_left[-approx_dist:][::-1])
            mr, br, mnr, bnr = getlinereg(x_smoothed_right[-approx_dist:][::-1], y_smoothed_right[-approx_dist:][::-1])
            
            #now try to find intersection of the normals with the opposite edge
            len_right_contour = len(right_contour)
            idx_right = len_right_contour-1
            #print(idx_right)
            if mnl == np.inf: #if we have a vertical normal line 
                for c in right_contour[:,0][::-1]: #goes backwards (starting at end point) through contour coordinates, c[1] = y, c[0] = x
                    if c[0] == x1_left: #check where our right coordinate has same x_value as left contour end point -> on vertical normal line
                        break 
                    idx_right  -= 1
                #cv.line(adgc, (x1_left, 0), (x1_left, img_height), (0,255,255), 1, cv.LINE_AA)
            elif mnl == 0: #if we have a horizontal normal line
                for c in right_contour[:,0][::-1]: #goes backwards (starting at end point) through contour coordinates, c[1] = y, c[0] = x
                    if c[1] == y1_left: #check where our right coordinate has same y_value as left contour end point -> on horizontal normal line
                        break 
                    idx_right  -= 1
                #cv.line(adgc, (0, y1_left), (img_width, y1_left), (0,255,255), 1, cv.LINE_AA)
            else: #if we don't have vertical nor horizontal line
                for c in right_contour[:,0][::-1]: #goes backwards (starting at end point) through contour coordinates, c[1] = y, c[0] = x
                    yc = round(mnl*c[0] +bnl) #plug into line equation, if it solves it it is on line 
                    if abs(yc-c[1]) <=5: #sometimes we have some rounding/ staircase error -> check if we are within reasonable range
                        break 
                    idx_right  -= 1 
                if idx_right < len_right_contour/2: #to catch unplausible cut, e.g. sharp drop form, normal bad approximated
                    idx_right = -1 


            #     #debug only
            #     y1line = round(bnl) 
            #     y2line = round(mnl*img_width+bnl)
            #     cv.line(adgc, (0, y1line), (img_width, y2line), (0,255,255), 1, cv.LINE_AA)
            
            # x_cross = right_contour[:,0,0][idx_right] 
            # y_cross = right_contour[:,0,1][idx_right]

            # cv.drawMarker(adgc, (x1_left, y1_left), color=(0,0,255), markerType = cv.MARKER_CROSS, markerSize=15, thickness=3)
            # cv.drawMarker(adgc, (x_cross, y_cross), color=(255,0,0), markerType = cv.MARKER_CROSS, markerSize=15, thickness=3)
            # cv.drawMarker(adgc, (x1_right, y1_right), color=(0,255,255), markerType = cv.MARKER_CROSS, markerSize=15, thickness=3)

            len_left_contour = len(left_contour)
            idx_left = len_left_contour-1
            #print(idx_left)
            if mnr == np.inf: #if we have a vertical normal line 
                for c in left_contour[:,0][::-1]: #goes backwards (starting at end point) through contour coordinates, c[1] = y, c[0] = x
                    if c[0] == x1_right: #check where our right coordinate has same x_value as left contour end point -> on vertical normal line
                        break 
                    idx_left  -= 1
                #cv.line(adgc, (x1_right, 0), (x1_right, img_height), (0,255,0), 1, cv.LINE_AA)
            elif mnr == 0: #if we have a horizontal normal line
                for c in left_contour[:,0][::-1]: #goes backwards (starting at end point) through contour coordinates, c[1] = y, c[0] = x
                    if c[1] == y1_right: #check where our right coordinate has same y_value as left contour end point -> on horizontal normal line
                        break 
                    idx_left  -= 1
                #cv.line(adgc, (0, y1_right), (img_width, y1_right), (0,255,0), 1, cv.LINE_AA)
            else: #if we don't have vertical nor horizontal line
                for c in left_contour[:,0][::-1]: #goes backwards (starting at end point) through contour coordinates, c[1] = y, c[0] = x
                    yc = round(mnr*c[0] +bnr)
                    if abs(yc-c[1]) <=5 : #sometimes we have some rounding/ staircase error -> check if we are within reasonable range
                        break
                    idx_left  -= 1
                if idx_left < len_left_contour/2: #to catch unplausible cut, e.g. sharp drop form, normal bad approximated
                    idx_left = -1

            #     #debug only
            #     y1line = round(bnr)
            #     y2line = round(mnr*img_width+bnr)
            #     cv.line(adgc, (0, y1line), (img_width, y2line), (0,255,0), 1, cv.LINE_AA)

            # x_cross = left_contour[:,0,0][idx_left]
            # y_cross = left_contour[:,0,1][idx_left]

            # cv.drawMarker(adgc, (x1_right, y1_right), color=(0,0,255), markerType = cv.MARKER_STAR, markerSize=15, thickness=3)
            # cv.drawMarker(adgc, (x_cross, y_cross), color=(255,0,0), markerType = cv.MARKER_STAR, markerSize=15, thickness=3)
            # cv.drawMarker(adgc, (x1_left, y1_left), color=(0,255,255), markerType = cv.MARKER_STAR, markerSize=15, thickness=3)


            #tv(adgc, size=0.5, time=1000)
            
            #cut the contours 
            left_contour = left_contour[:idx_left]
            right_contour = right_contour[:idx_right]

            # lets do a final smoothing run on the final cutted contours 
            left_bspline_points = int(len(left_contour[:,0,1])) #will give us the same number of points as the raw contours
            right_bspline_points = int(len(right_contour[:,0,1]))

            #array shaping should be done better
            left_contour_l1 = left_contour[:,0,1][0::10]
            left_contour_l0 = left_contour[:,0,0][0::10]
            #make sure last point is in shortened contour list, sometimes not the case as number of points not always dividable by 10 withour rest 
            if left_contour_l1[-1] != left_contour[:,0,1][-1] or left_contour_l0[-1] != left_contour[:,0,0][-1]:
                left_contour_l1 = np.append(left_contour_l1, left_contour[:,0,1][-1])
                left_contour_l0= np.append(left_contour_l0, left_contour[:,0,0][-1])

            left_contour_l = (left_contour_l1, left_contour_l0) #(y, x)
            
            o = []
            for i in range(len(left_contour_l[0])):
                c = [left_contour_l[1][i],left_contour_l[0][i]]
                o.append(c)
            
            left_contour_smoothed = scipy_bspline(o,left_bspline_points,degree=5) #degree 5 seems to work well
            x_smoothed_left = left_contour_smoothed[:,0].astype('int')
            y_smoothed_left = left_contour_smoothed[:,1].astype('int')


            right_contour_l1 = right_contour[:,0,1][0::10]
            right_contour_l0 = right_contour[:,0,0][0::10]
            #make sure last point is in shortened contour list, sometimes not the case as number of points not always dividable by 10 withour rest 
            if right_contour_l1[-1] != right_contour[:,0,1][-1] or right_contour_l0[-1] != right_contour[:,0,0][-1]:
                right_contour_l1 = np.append(right_contour_l1, right_contour[:,0,1][-1])
                right_contour_l0= np.append(right_contour_l0, right_contour[:,0,0][-1])

            right_contour_l = (right_contour_l1, right_contour_l0) #(y, x)

            o = []
            for i in range(len(right_contour_l[0])):
                c = [right_contour_l[1][i],right_contour_l[0][i]]
                o.append(c)

            right_contour_smoothed = scipy_bspline(o,right_bspline_points,degree=5)
            x_smoothed_right = right_contour_smoothed[:,0].astype('int')
            y_smoothed_right = right_contour_smoothed[:,1].astype('int')


            #lets bring our contour data in shape
            left_contour = (left_contour[:,0,1], left_contour[:,0,0]) #(y, x)
            right_contour = (right_contour[:,0,1], right_contour[:,0,0])
            left_contour_smoothed = (y_smoothed_left, x_smoothed_left)
            right_contour_smoothed = (y_smoothed_right, x_smoothed_right)
    except:
        adgimage = image
        status = "Contours not found - exception occured"
        left_contour_smoothed = 0
        right_contour_smoothed = 0
        left_contour = 0
        right_contour = 0
        ll_limit = 0
        lr_limit = 0
        rl_limit = 0
        rr_limit = 0
        all_contours = 0
        contours = 0

    return adgimage, status,left_contour_smoothed, right_contour_smoothed, left_contour, right_contour, ll_limit, lr_limit, rl_limit, rr_limit, all_contours, contours  

def getJetContoursHorizontalLine(image, collector, nozzleleftlowy, nozzlerightlowy, nozzleleftlowx, nozzlerightlowx, Threshold_kernel=101, Blur_kernel=3, upper_cut_off=0, lower_cut_off=1000):
    try:
        
        #optionaly implement gaussian blur before edge detection to reduce noise if needed
        if Blur_kernel != 0:
            image = cv.GaussianBlur(image,(Blur_kernel,Blur_kernel),0)


        img_width = image.shape[1]
        img_height = image.shape[0]
        imgedge_pixel_dist = 10
        #image = cv.bitwise_not(image) #test for cheap cam
        #21 kernel for JC1 sometimes ok, 101 seems saver overall
        adgimage = cv.adaptiveThreshold(image,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,Threshold_kernel,2) #101 kernel seems to work fine with low contrast images #better than mean
        #tv(adgimage, size=0.5)
        adgimage[:upper_cut_off,:] = 0 #set everything above upper cut off black -> so we won't find any contours there
        adgimage[:,:imgedge_pixel_dist] = 0 #set 10 pixel from left edge black, so we dont run into image edge
        adgimage[:,img_width-imgedge_pixel_dist:] = 0 #set 10 pixel from right edge black, so we dont run into image edge
        kernel = np.ones((3,3),np.uint8)
        adgimage = cv.morphologyEx(adgimage, cv.MORPH_OPEN, kernel, iterations=2) #remove some of the noise specs 

        #tv(adgimage, size=0.5)
        #adgc = cv.cvtColor(adgimage, cv.COLOR_GRAY2BGR) #for debug only
        
        #fill image black below collector line to get rid of reflections
        adgimage[collector:,:] = 0
        #tv(adgimage, size=0.5)

        #horizontal line should use other reference, for now
        collector2 = np.rint(collector/2)
        


        #tv(adgimage, size=0.5)
        #some variables for hough line detection, selected by trial and error 
        HLrho=1
        HLtheta=(np.pi/180)
        HLthreshold=50
        HLminLineLength=300 #was 100
        HLmaxLineGap=500 #was 500
        HorizontalAngleThres=3 # throw away lines that are not almost horizontal
        lines = cv.HoughLinesP(adgimage, rho=HLrho, theta=HLtheta, threshold=HLthreshold, minLineLength=HLminLineLength, maxLineGap=HLmaxLineGap) 
        if lines is not None: 
            horizontal_lines = [] #create a list for horizontal lines
            for i in range(len(lines)): #go through the list of lines
                l = lines[i][0] #pick the vector that represents (x0,y0,x1,y1), for some reason lines structure is [[[x0.0 y0.0 x1.0 y1.0]][[x0.1 y0.1 x1.1 y1.1]]] .0 first line, .1 second line...
                
                #cv.line(adgc, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv.LINE_AA) #draw all lines, for debugging
                #cv.drawMarker(adgc, (l[0], l[1]), (255,0,0), markerType = cv.MARKER_SQUARE, markerSize= 20) #should be top, debug only
                #cv.drawMarker(adgc, (l[2], l[3]), (0,255,0), markerType = cv.MARKER_TILTED_CROSS, markerSize= 20) #should be bottom, debug only
                #tv(adgc, size=0.5)

                if abs(np.arctan2((l[3]-l[1]),(l[2]-l[0])) *180 /np.pi  ) < HorizontalAngleThres and l[1] > collector2 and l[3] > collector2: #if the angle of the line is greater horizontal_angle_thres, -> look for (almost) horizontal lines
                    if l[0] > l[1]: #if the first point further left than the second point, swap them. Sorts the line points so that first point is always on the left 
                        lines[i][0] = [l[2], l[3], l[0], l[1]] 

                    horizontal_lines.append(l) #append line to horizontal line list 
                    #print(np.arctan2((l[3]-l[1]),(l[2]-l[0])) *180 /np.pi) #debug only
                    #cv.line(timg, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv.LINE_AA) #draw the vertical line, debugging only
                    #tv(timg) #debuggin only
            
            #tv(adgc, size=0.5)
            y_min = img_height #init with lower image edge
            if horizontal_lines is not None:
                for i in range(len(horizontal_lines)): #go through horizontal lines, and find upmost y_value in start and end points (left and right points)
                    l = horizontal_lines[i] #pick the vector that represents (x0,y0,x1,y1)
                    y_min = min(l[1],l[3], y_min)
                    #cv.line(adgc, (0, y_min), (img_width, y_min), (0,0,255), 1, cv.LINE_AA)
            
                adgimage[y_min:,:] = 0 #=> horizontal line avoids ambiguity in lowest position of contours
            else:
                adgimage[imgedge_pixel_dist:,:] = 0

        adgimage = cv.morphologyEx(adgimage, cv.MORPH_CLOSE, kernel, iterations=3)

        contours, hierarchy = cv.findContours(adgimage, mode = cv.RETR_TREE, method = cv.CHAIN_APPROX_NONE) #joining continuous points having same color/intesity, returns a list of the contours, represented as x/y coordinate arrays. 
        all_contours = contours
        #RETR_LIST returns all contours, method to apporxe_none to store all contour points
        #RETR_EXTERNAL, returns only outer extreme contours
        #RETR_TREE gives full hierarchy, needed so we can detect if inner or outer contour => clockwise or counterclockwise point array

        #print(contours)
        #contours = sorted(contours, key=lambda x: (cv.boundingRect(x))[1])#sort contours increasing y-value upper left point of bouning rectangle boundingRect returns: top left point x, y, width, heigth
        #New version with hiearchy:
        sort_idx = list(range(len(contours))) #create a list with 0,1,2,3..... for length of contours
        s = sorted(zip(contours,sort_idx), key=lambda x: (cv.boundingRect(x[0])[2]*cv.boundingRect(x[0])[3]), reverse=True)#sort contours decreasing y-value upper left point of bouning rectangle boundingRect returns: top left point x, y, width, heigth
        contours, sort_idx = zip(*s) #unzip contours and sort_idx, sort_idx is sorted like contours, so it gives us new index order
        contours = list(contours) #turn tuple into list type for following sorting
        sort_idx = np.array(sort_idx) #turn it into numpy array
        hierarchy[0][sort_idx] #use sort_idx to sort hierarchy in same way our contours are now sorted
        hierarchy = list(hierarchy[0]) #turn into list of arrays, so we can also sort it in the following

        remove_idx = []
        nozzle_dx_1_4 = abs(nozzleleftlowx - nozzlerightlowx)/4
        ll_limit = nozzleleftlowx - nozzle_dx_1_4   #   | |--nozzle--| |
        lr_limit = nozzleleftlowx + nozzle_dx_1_4   #   | |--nozzle--| |
        rl_limit = nozzlerightlowx - nozzle_dx_1_4  #   | |--nozzle--| |
        rr_limit = nozzlerightlowx + nozzle_dx_1_4  #   | | |      | | |
                                                    #  ll   lr     rl  rr 

        for i in range(len(contours)): #remove contours based on some criteria 
            #print(i)
            #print(cv.boundingRect(contours[i])[1])
            #boundingRect returns: top left point x, y, width, heigth
            x_up_left = cv.boundingRect(contours[i])[0]
            y_up_left = cv.boundingRect(contours[i])[1]
            box_width = cv.boundingRect(contours[i])[2]
            box_height = cv.boundingRect(contours[i])[3]
            #print("contour area:", box_width*box_height)
            #nozzleleftlowx, nozzlerightlowx
            if y_up_left + box_height < upper_cut_off: #if the lower bounding box edge is above our cut_off => remove it from list, is fragmented nozzle contour
                remove_idx.append(i) 
            elif y_up_left > lower_cut_off: # if the contours upper left point starts below the lower cut off, it is not attached to nozzle/starts below it, remove it. could e.g. be a contour inside the jet/blob
                remove_idx.append(i)
            #check if the upper left or right corner of the bounding box is within 1/4 width of nozzle from left/right lower nozzle edge, in x-direction
            # this removes contours that dont't start close to nozzle 
            elif not ( (not (ll_limit<x_up_left<lr_limit) and not(rl_limit<(x_up_left+box_width)<rr_limit )) or (not (ll_limit<x_up_left+box_width<lr_limit) and not(rl_limit<x_up_left<rr_limit ))):
                remove_idx.append(i)
            
        if len(remove_idx) != 0:
            for index in sorted(remove_idx, reverse=True):
                del contours[index]
                del hierarchy[index]
        
        #box_width = cv.boundingRect(contours[0])[2]
        #box_height = cv.boundingRect(contours[0])[3]
        #print("remaining contour area:", box_width*box_height)
        #print(type(contours)) #debug only
        #print(contours) #debug only
        #print("number of contours:", len(contours))
        #print(len(hierarchy))



        if len(contours) == 0:
            status = "no valid contour found"
            left_contour = 0
            right_contour = 0

        else: #always pick first contour, should be the biggest one (sometimes we detect almost the same contour multiple times)
        #print("handling one contour case")
            status = "Ok"
            contour_direction = hierarchy[0][3]
            #print("hieararchy", contour_direction)
            single_contour = contours[0]
            #contour direction => -1 = counter-clockwise, != -1 => clockwise
            if contour_direction != -1: #if order of coordiante points is not counter clockwise, we turn it around
                single_contour = single_contour[::-1]
                #print("flipped direction")

            #print(single_contour)
            #print(single_contour[len(single_contour)//2])
            x_start = single_contour[0,0,0] 
            #check if our contour start point is on left or right side of nozzle, changes depending on how high the contour goes on each side (higher side is start side)
            if (ll_limit<x_start<lr_limit): 
                start_side = 0
                #print("start left")
                #get list of indices where our x value is in the range accepted around the right nozzle edge
                id_other_side_x = np.where(np.logical_and(single_contour[:,0,0]>rl_limit, single_contour[:,0,0]<rr_limit))[0]
                #search in that indices range for the lowest y_value => highest point
                y_max_other_side = np.where(single_contour[id_other_side_x[0]:id_other_side_x[-1],0,1] == np.min(single_contour[id_other_side_x[0]:id_other_side_x[-1],0,1]) )[0][0]
                #calculate the index of that point, by adding the offset created by the list 
                id_top_other_side = id_other_side_x[0]+y_max_other_side
            else:
                start_side = 1
            # print("start right")
                #since we are on right side, we have to flip it around to go clock wise, thus along lower side of contour 
                single_contour = single_contour[::-1] 
                #get list of indices where our x value is in the range accepted around the left nozzle edge
                id_other_side_x = np.where(np.logical_and(single_contour[:,0,0]>ll_limit, single_contour[:,0,0]<lr_limit))[0]
                #search in that indices range for the lowest y_value => highest point
                y_max_other_side = np.where(single_contour[id_other_side_x[0]:id_other_side_x[-1],0,1] == np.min(single_contour[id_other_side_x[0]:id_other_side_x[-1],0,1]) )[0][0]
                #calculate the index of that point, by adding the offset created by the list 
                id_top_other_side = id_other_side_x[0]+y_max_other_side
                    
            single_contour = single_contour[:id_top_other_side] #take only part of contour until top of other side, after that the contour wraps around/runs back on inside of edge, we dont need that part
            
            y_max = np.max(single_contour[:,0,1]) #get max y_value
            idx_max_y = np.where(single_contour[:,0,1] == y_max)[0] #get list of index where our y is equal y_max

            #cv.line(adgc, (0, y_max), (img_width, y_max), (255,0,0), 1, cv.LINE_AA) #debug only
            #tv(adgc, size = 0.5)
            
            #print("idx_max_y", idx_max_y)
            if len(idx_max_y) == 1:
                idx_max_y_left = idx_max_y[0]
                idx_max_y_right = idx_max_y[0]
            else:
                if start_side == 0: # in this case we walk around counter clock wise, so first entry we find in list belongs to left side
                    idx_max_y_left = idx_max_y[0]
                    idx_max_y_right = idx_max_y[-1]
                else: # in this case we walk around clock wise, so first entry we find in list belongs to right side
                    idx_max_y_left = idx_max_y[-1]
                    idx_max_y_right = idx_max_y[0]
            

            #check if we run into image edge, if so, don't split contour according to y_max, but along edge
            idx_imgedge_x = np.where(single_contour[:,0,0] == imgedge_pixel_dist)[0] #get list of index where our x is equal 0, left image edge
            if len(idx_imgedge_x) == 0: 
                idx_imgedge_x = np.where(single_contour[:,0,0] == img_width-imgedge_pixel_dist)[0] #get list of index where our x is equal img_width, right image edge

            # #print(idx_imgedge_x)
            k = len(idx_imgedge_x)
            if k != 0:
                if start_side == 0:
                    idx_max_y_left = idx_imgedge_x[0]
                    idx_max_y_right = idx_imgedge_x[-1]
                else: # in this case we walk around clock wise, so first entry we find in list belongs to right side
                    idx_max_y_left = idx_imgedge_x[-1]
                    idx_max_y_right = idx_imgedge_x[0]

            #old
            #idx_max_y_mid = idx_max_y[len(idx_max_y)//2] #get index at middle of list , e.g. we could have more than one pixel at y_max, we want to break apart at middle of it
            #print("mid index", idx_max_y_mid, "of", len(idx_max_y))
            
            #if we start on left side, our coordinates are sorted counter clockwise
            if start_side == 0:
                left_contour = single_contour[:idx_max_y_left] #create left part of contour 
                right_contour = single_contour[idx_max_y_right:][::-1] #create right part of contour, flip its direction to have the normal start at nozzle orientation
            #if we start on right side, our coordinates are sorted clockwise
            else:
                left_contour = single_contour[idx_max_y_left:][::-1] #create left part of contour 
                right_contour = single_contour[:idx_max_y_right] #create right part of contour, flip its direction to have the normal start at nozzle orientation
                
        
            #Calculate m and b for lower nozzle edge  #points are [[xhigh,xlow],[yhigh,ylow]] 
            if (nozzlerightlowx-nozzleleftlowx) != 0:  
                m = (nozzlerightlowy-nozzleleftlowy)/(nozzlerightlowx-nozzleleftlowy) # m = (y1-y0)/(x1-x0)
                b = nozzlerightlowy-m*nozzlerightlowx #b = yi-m*xi

            #we just plug in the edge points into our line definition and check if it is solution, if so we found the intersection point
            u_idx_left = 0
            for c in left_contour[:,0]:
                if c[1] == round(m*c[0] +b): #round() slighly faster than int(np.rint)
                    break 
                u_idx_left  += 1 

            u_idx_right = 0
            for c in right_contour[:,0]:
                if c[1] == round(m*c[0] +b):
                    break 
                u_idx_right  += 1
            
            left_contour = left_contour[u_idx_left:]
            right_contour = right_contour[u_idx_right:]

            
            #Now lets try to cut the end of the jet in a perpendicular manner, 
            # get the normal for left and right contour.
            #approx short end section of contour with a line.

            #first approx the left/right contour with a bspline to make it a bit smoother
            #number of points to create out of bspline, this will be the moothed contour
            left_bspline_points = int(len(left_contour[:,0,1])/10) #1/10 of original number of points seems to work fine
            right_bspline_points = int(len(right_contour[:,0,1])/10)
            #array shaping should be done better
            left_contour_l = (left_contour[:,0,1], left_contour[:,0,0]) #(y, x)
            o = []
            for i in range(len(left_contour_l[0])):
                c = [left_contour_l[1][i],left_contour_l[0][i]]
                o.append(c)
            
            left_contour_smoothed = scipy_bspline(o,left_bspline_points,degree=5) #degree 5 seems to work well
            x_smoothed_left = left_contour_smoothed[:,0].astype('int')
            y_smoothed_left = left_contour_smoothed[:,1].astype('int')

            right_contour_l = (right_contour[:,0,1], right_contour[:,0,0]) #(y, x)
            o = []
            for i in range(len(right_contour_l[0])):
                c = [right_contour_l[1][i],right_contour_l[0][i]]
                o.append(c)

            right_contour_smoothed = scipy_bspline(o,right_bspline_points,degree=5)
            x_smoothed_right = right_contour_smoothed[:,0].astype('int')
            y_smoothed_right = right_contour_smoothed[:,1].astype('int')

            #lower end points of our smoothed contours, should be equal the original contour (bspline runs through end points)
            x1_left = x_smoothed_left[-1] #end point x, left contour
            y1_left = y_smoothed_left[-1] #end point y, right contour
            x1_right = x_smoothed_right[-1] #end point x, right contour
            y1_right = y_smoothed_right[-1] #end point y, right contour

            # how many of our smoothed points to use to approximate slope of end section
            approx_dist = 5 #five seems to work fine 
            #get the slope and offset of the end section and of the normal 
            ml, bl, mnl, bnl = getlinereg(x_smoothed_left[-approx_dist:][::-1], y_smoothed_left[-approx_dist:][::-1])
            mr, br, mnr, bnr = getlinereg(x_smoothed_right[-approx_dist:][::-1], y_smoothed_right[-approx_dist:][::-1])
            
            #now try to find intersection of the normals with the opposite edge
            len_right_contour = len(right_contour)
            idx_right = len_right_contour-1
            #print(idx_right)
            if mnl == np.inf: #if we have a vertical normal line 
                for c in right_contour[:,0][::-1]: #goes backwards (starting at end point) through contour coordinates, c[1] = y, c[0] = x
                    if c[0] == x1_left: #check where our right coordinate has same x_value as left contour end point -> on vertical normal line
                        break 
                    idx_right  -= 1
                #cv.line(adgc, (x1_left, 0), (x1_left, img_height), (0,255,255), 1, cv.LINE_AA)
            elif mnl == 0: #if we have a horizontal normal line
                for c in right_contour[:,0][::-1]: #goes backwards (starting at end point) through contour coordinates, c[1] = y, c[0] = x
                    if c[1] == y1_left: #check where our right coordinate has same y_value as left contour end point -> on horizontal normal line
                        break 
                    idx_right  -= 1
                #cv.line(adgc, (0, y1_left), (img_width, y1_left), (0,255,255), 1, cv.LINE_AA)
            else: #if we don't have vertical nor horizontal line
                for c in right_contour[:,0][::-1]: #goes backwards (starting at end point) through contour coordinates, c[1] = y, c[0] = x
                    yc = round(mnl*c[0] +bnl) #plug into line equation, if it solves it it is on line 
                    if abs(yc-c[1]) <=5: #sometimes we have some rounding/ staircase error -> check if we are within reasonable range
                        break 
                    idx_right  -= 1 
                if idx_right < len_right_contour/2: #to catch unplausible cut, e.g. sharp drop form, normal bad approximated
                    idx_right = -1 


            #     #debug only
            #     y1line = round(bnl) 
            #     y2line = round(mnl*img_width+bnl)
            #     cv.line(adgc, (0, y1line), (img_width, y2line), (0,255,255), 1, cv.LINE_AA)
            
            # x_cross = right_contour[:,0,0][idx_right] 
            # y_cross = right_contour[:,0,1][idx_right]

            # cv.drawMarker(adgc, (x1_left, y1_left), color=(0,0,255), markerType = cv.MARKER_CROSS, markerSize=15, thickness=3)
            # cv.drawMarker(adgc, (x_cross, y_cross), color=(255,0,0), markerType = cv.MARKER_CROSS, markerSize=15, thickness=3)
            # cv.drawMarker(adgc, (x1_right, y1_right), color=(0,255,255), markerType = cv.MARKER_CROSS, markerSize=15, thickness=3)

            len_left_contour = len(left_contour)
            idx_left = len_left_contour-1
            #print(idx_left)
            if mnr == np.inf: #if we have a vertical normal line 
                for c in left_contour[:,0][::-1]: #goes backwards (starting at end point) through contour coordinates, c[1] = y, c[0] = x
                    if c[0] == x1_right: #check where our right coordinate has same x_value as left contour end point -> on vertical normal line
                        break 
                    idx_left  -= 1
                #cv.line(adgc, (x1_right, 0), (x1_right, img_height), (0,255,0), 1, cv.LINE_AA)
            elif mnr == 0: #if we have a horizontal normal line
                for c in left_contour[:,0][::-1]: #goes backwards (starting at end point) through contour coordinates, c[1] = y, c[0] = x
                    if c[1] == y1_right: #check where our right coordinate has same y_value as left contour end point -> on horizontal normal line
                        break 
                    idx_left  -= 1
                #cv.line(adgc, (0, y1_right), (img_width, y1_right), (0,255,0), 1, cv.LINE_AA)
            else: #if we don't have vertical nor horizontal line
                for c in left_contour[:,0][::-1]: #goes backwards (starting at end point) through contour coordinates, c[1] = y, c[0] = x
                    yc = round(mnr*c[0] +bnr)
                    if abs(yc-c[1]) <=5 : #sometimes we have some rounding/ staircase error -> check if we are within reasonable range
                        break
                    idx_left  -= 1
                if idx_left < len_left_contour/2: #to catch unplausible cut, e.g. sharp drop form, normal bad approximated
                    idx_left = -1

            #     #debug only
            #     y1line = round(bnr)
            #     y2line = round(mnr*img_width+bnr)
            #     cv.line(adgc, (0, y1line), (img_width, y2line), (0,255,0), 1, cv.LINE_AA)

            # x_cross = left_contour[:,0,0][idx_left]
            # y_cross = left_contour[:,0,1][idx_left]

            # cv.drawMarker(adgc, (x1_right, y1_right), color=(0,0,255), markerType = cv.MARKER_STAR, markerSize=15, thickness=3)
            # cv.drawMarker(adgc, (x_cross, y_cross), color=(255,0,0), markerType = cv.MARKER_STAR, markerSize=15, thickness=3)
            # cv.drawMarker(adgc, (x1_left, y1_left), color=(0,255,255), markerType = cv.MARKER_STAR, markerSize=15, thickness=3)


            #tv(adgc, size=0.5, time=1000)
            
            #cut the contours 
            left_contour = left_contour[:idx_left]
            right_contour = right_contour[:idx_right]

            # lets do a final smoothing run on the final cutted contours 
            left_bspline_points = int(len(left_contour[:,0,1])) #will give us the same number of points as the raw contours
            right_bspline_points = int(len(right_contour[:,0,1]))

            #array shaping should be done better
            left_contour_l1 = left_contour[:,0,1][0::10]
            left_contour_l0 = left_contour[:,0,0][0::10]
            #make sure last point is in shortened contour list, sometimes not the case as number of points not always dividable by 10 withour rest 
            if left_contour_l1[-1] != left_contour[:,0,1][-1] or left_contour_l0[-1] != left_contour[:,0,0][-1]:
                left_contour_l1 = np.append(left_contour_l1, left_contour[:,0,1][-1])
                left_contour_l0= np.append(left_contour_l0, left_contour[:,0,0][-1])

            left_contour_l = (left_contour_l1, left_contour_l0) #(y, x)
            
            o = []
            for i in range(len(left_contour_l[0])):
                c = [left_contour_l[1][i],left_contour_l[0][i]]
                o.append(c)
            
            left_contour_smoothed = scipy_bspline(o,left_bspline_points,degree=5) #degree 5 seems to work well
            x_smoothed_left = left_contour_smoothed[:,0].astype('int')
            y_smoothed_left = left_contour_smoothed[:,1].astype('int')


            right_contour_l1 = right_contour[:,0,1][0::10]
            right_contour_l0 = right_contour[:,0,0][0::10]
            #make sure last point is in shortened contour list, sometimes not the case as number of points not always dividable by 10 withour rest 
            if right_contour_l1[-1] != right_contour[:,0,1][-1] or right_contour_l0[-1] != right_contour[:,0,0][-1]:
                right_contour_l1 = np.append(right_contour_l1, right_contour[:,0,1][-1])
                right_contour_l0= np.append(right_contour_l0, right_contour[:,0,0][-1])

            right_contour_l = (right_contour_l1, right_contour_l0) #(y, x)

            o = []
            for i in range(len(right_contour_l[0])):
                c = [right_contour_l[1][i],right_contour_l[0][i]]
                o.append(c)

            right_contour_smoothed = scipy_bspline(o,right_bspline_points,degree=5)
            x_smoothed_right = right_contour_smoothed[:,0].astype('int')
            y_smoothed_right = right_contour_smoothed[:,1].astype('int')


            #lets bring our contour data in shape
            left_contour = (left_contour[:,0,1], left_contour[:,0,0]) #(y, x)
            right_contour = (right_contour[:,0,1], right_contour[:,0,0])
            left_contour_smoothed = (y_smoothed_left, x_smoothed_left)
            right_contour_smoothed = (y_smoothed_right, x_smoothed_right)
    except:
        adgimage = image
        status = "Contours not found - exception occured"
        left_contour_smoothed = 0
        right_contour_smoothed = 0
        left_contour = 0
        right_contour = 0
        ll_limit = 0
        lr_limit = 0
        rl_limit = 0
        rr_limit = 0
        all_contours = 0
        contours = 0
        y_min = 0


    return adgimage, status,left_contour_smoothed, right_contour_smoothed, left_contour, right_contour, ll_limit, lr_limit, rl_limit, rr_limit, all_contours, contours, y_min


def getJetContoursHorizontalLineScan(image, collector_left, collector_right, nozzleleftlowy, nozzlerightlowy, nozzleleftlowx, nozzlerightlowx, Threshold_kernel=101, Blur_kernel=3, upper_cut_off=0, lower_cut_off=1000):
    try:
        #optionaly implement gaussian blur before edge detection to reduce noise if needed
        if Blur_kernel != 0:
            image = cv.GaussianBlur(image,(Blur_kernel,Blur_kernel),0)


        img_width = image.shape[1]
        img_height = image.shape[0]
        imgedge_pixel_dist = 10 #10
        #image = cv.bitwise_not(image) #test for cheap cam
        #21 kernel for JC1 sometimes ok, 101 seems saver overall
        adgimage = cv.adaptiveThreshold(image,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,Threshold_kernel,2) #101 kernel seems to work fine with low contrast images #better than mean
        #print("raw image")
        #tv(adgimage, size=0.5)
        adgimage[:upper_cut_off,:] = 0 #set everything above upper cut off black -> so we won't find any contours there
        adgimage[:,:imgedge_pixel_dist] = 0 #set some pixel from left edge black, so we dont run into image edge
        adgimage[:,img_width-imgedge_pixel_dist:] = 0 #set some pixel from right edge black, so we dont run into image edge
        #polygon for area below collector
        points = np.array([[0,collector_left],[0,img_height],[img_width,img_height],[img_width,collector_right]])
        cv.fillPoly(adgimage,[points],color=0)
        #print("after fillpoly - before first morph close")
        #tv(adgimage, size=0.5)
        kernel = np.ones((3,3),np.uint8)
        adgimage = cv.morphologyEx(adgimage, cv.MORPH_OPEN, kernel, iterations=2) #remove some of the noise specs 
        #print("after first morph close - before linecan")
        #tv(adgimage, size=0.5)
        #adgc = cv.cvtColor(adgimage, cv.COLOR_GRAY2BGR) #for debug only
        
        #fill image black below collector line to get rid of reflections
        #adgimage[collector:,:] = 0 #redundant with code below
        #tv(adgimage, size=0.5)

        #horizontal line should use other reference, for now
        max_collector = max(collector_left, collector_right)
        collector2 = int(np.rint(max_collector/2))
        y_min_l = collector_left
        y_min_r = collector_right

        #while True:
        y_min = max_collector
        try:
           # counts = np.zeros(collector-(collector2+1))
            line = adgimage[collector2]
            nonz = np.nonzero(line)
            last_count = np.count_nonzero(line) #faster then len(np.nonzero[0])
       
            xl = np.int64(max(nonz[0][0] - 100, 0)) #weird issue if result is 0, need to specify type np.int64, otherwise  nonz+xl fails for xl=0
            xr = np.int64(min(nonz[0][-1] + 100, img_width-1))
        
            #print(xl, xr)
            # print(nonz)

            # timg = image.copy()
            # cv.line(timg, (xl, collector-100), (xr, collector-100), 100, 3, cv.LINE_AA) #draw all lines, for debugging
            # cv.line(timg, (nonz[0][0], collector-100), (nonz[0][-1], collector-100), 255, 2, cv.LINE_AA) #draw all lines, for debugging
            # tv(timg, size = 0.5)
            #print("before for-loophere")
            draw_img = False
            for i in range (collector2+1,max_collector-1,1):
               # print(i)
                line = adgimage[i,xl:xr]
                #print("got line from image")
                #print("line:", line)
                nonz = np.nonzero(line) + xl
                #print("got nonzero from line")
                #print("nonz+xl:", nonz)
                nonzcount = np.count_nonzero(line)
                #print("got nonzero count")
                #print("nonzcoutn:", nonzcount)


                #old - actually we don't need connected regions, total count of pizels should be enough 
                # taken from: https://stackoverflow.com/questions/24342047/count-consecutive-occurences-of-values-varying-in-length-in-a-numpy-array
                #connected_counts = np.diff(np.where(np.concatenate(([line[0]],line[:-1] != line[1:],[255])))[0])[::2]
                # gives us a list with length equal to the number of connected regions and each entry equal to the number of pixels in the connected region
                #nonz = np.where(line)
                #nonz = nonz +xl
                if nonzcount == 0: #may happen if we the jet is not relatively hoizontal and not touching the collector -> almost horizontal => not enough new pixels to trigger nonzcount-last_count => we reach gap between jet and its reflection => all black pixels => would fail nonz[0][0] access 
                    y_min = i
                    draw_img = True
                    #print("nonzcount case")
                    break
                else:
                    xl = np.int64(max(nonz[0][0] - 100, 0)) #weird issue if result is 0, need to specify type np.int64, otherwise  nonz+xl fails for xl=0
                    xr = np.int64(min(nonz[0][-1] + 100, img_width-1))
                    #print("got xl and xr")
                    #print("xl:", xl, "type:", type(xl))
                    #print("xr:", xr, "type:", type(xr))
                #print("y:", i, "collector y:", collector)
                #print(max(connected_counts))
                #print(adgimage[i,xl:xr])
               # print(nonzcount)
                #print(nonzcount-last_count)

                if abs(nonzcount-last_count) > 30: #if at least 30 more pixels are white in current line -> we reach into deposited fiber - should be more adaptive 
                    y_min = i #current i = y_min where we should cut off #-5 #maybe adjust distance a bit 10 pixels? 
                    draw_img = True
                    #print("delta nonz > 30 case")
                    #print(y_min)
                    break # even if we never reach this condition, i will run to collector value -> cut off of image below collector line should always be ensured
                last_count = nonzcount

            if draw_img == True:
                adgimage[y_min:,:] = 0
                y_min_l = y_min
                y_min_r = y_min
                #print("linescan draw")
           #print(y_min)


                # if nonzcount > 100: #maybe find better measure -> diff to previous?
                #     y_min = i-5 #maybe adjust distance a bit 10 pixels?
                #     print(y_min)
                #     adgimage[y_min:,:] = 0
                #     break

                # timg = image.copy()
                # cv.line(timg, (xl, i), (xr, i), 100, 3, cv.LINE_AA) #draw all lines, for debugging
                # cv.line(timg, (nonz[0][0], i), (nonz[0][-1], i), 255, 2, cv.LINE_AA) #draw all lines, for debugging
                # tv(timg, size = 0.5)
                #print(connected_counts)



        except:
            #print("exception line scan")
            pass
        #print("after linescan - before second morph close")
        #tv(adgimage, size=0.5)
 
        adgimage = cv.morphologyEx(adgimage, cv.MORPH_CLOSE, kernel, iterations=10) #important to close shine-through gap in jet 
        #print("after second morph close - before contour")
        #tv(adgimage, size=0.5)

        #cv.imwrite(r"E:\Figure_1_SelectedIMages\Pulsing\Output\AnnotatedImages\CV_threshold_image20600.jpg", adgimage)

        contours, hierarchy = cv.findContours(adgimage, mode = cv.RETR_TREE, method = cv.CHAIN_APPROX_NONE) #joining continuous points having same color/intesity, returns a list of the contours, represented as x/y coordinate arrays. 
        all_contours = contours
        #RETR_LIST returns all contours, method to apporxe_none to store all contour points
        #RETR_EXTERNAL, returns only outer extreme contours
        #RETR_TREE gives full hierarchy, needed so we can detect if inner or outer contour => clockwise or counterclockwise point array

        #print(contours)
        #contours = sorted(contours, key=lambda x: (cv.boundingRect(x))[1])#sort contours increasing y-value upper left point of bouning rectangle boundingRect returns: top left point x, y, width, heigth
        #New version with hiearchy:
        sort_idx = list(range(len(contours))) #create a list with 0,1,2,3..... for length of contours
        s = sorted(zip(contours,sort_idx), key=lambda x: (cv.boundingRect(x[0])[2]*cv.boundingRect(x[0])[3]), reverse=True)#sort contours decreasing y-value upper left point of bouning rectangle boundingRect returns: top left point x, y, width, heigth
        contours, sort_idx = zip(*s) #unzip contours and sort_idx, sort_idx is sorted like contours, so it gives us new index order
        contours = list(contours) #turn tuple into list type for following sorting
        sort_idx = np.array(sort_idx) #turn it into numpy array
        hierarchy[0][sort_idx] #use sort_idx to sort hierarchy in same way our contours are now sorted
        hierarchy = list(hierarchy[0]) #turn into list of arrays, so we can also sort it in the following

        remove_idx = []
        nozzle_dx_1_4 = abs(nozzleleftlowx - nozzlerightlowx)/4
        ll_limit = nozzleleftlowx - nozzle_dx_1_4   #   | |--nozzle--| |
        lr_limit = nozzleleftlowx + nozzle_dx_1_4   #   | |--nozzle--| |
        rl_limit = nozzlerightlowx - nozzle_dx_1_4  #   | |--nozzle--| |
        rr_limit = nozzlerightlowx + nozzle_dx_1_4  #   | | |      | | |
                                                    #  ll   lr     rl  rr 

        for i in range(len(contours)): #remove contours based on some criteria 
            #print(i)
            #print(cv.boundingRect(contours[i])[1])
            #boundingRect returns: top left point x, y, width, heigth
            x_up_left = cv.boundingRect(contours[i])[0]
            y_up_left = cv.boundingRect(contours[i])[1]
            box_width = cv.boundingRect(contours[i])[2]
            box_height = cv.boundingRect(contours[i])[3]
            #print("contour area:", box_width*box_height)
            #nozzleleftlowx, nozzlerightlowx
            if y_up_left + box_height < upper_cut_off: #if the lower bounding box edge is above our cut_off => remove it from list, is fragmented nozzle contour
                remove_idx.append(i) 
            elif y_up_left > lower_cut_off: # if the contours upper left point starts below the lower cut off, it is not attached to nozzle/starts below it, remove it. could e.g. be a contour inside the jet/blob
                remove_idx.append(i)
            #check if the upper left or right corner of the bounding box is within 1/4 width of nozzle from left/right lower nozzle edge, in x-direction
            # this removes contours that dont't start close to nozzle 
            elif not ( (not (ll_limit<x_up_left<lr_limit) and not(rl_limit<(x_up_left+box_width)<rr_limit )) or (not (ll_limit<x_up_left+box_width<lr_limit) and not(rl_limit<x_up_left<rr_limit ))):
                remove_idx.append(i)
            
        if len(remove_idx) != 0:
            for index in sorted(remove_idx, reverse=True):
                del contours[index]
                del hierarchy[index]
        
        #box_width = cv.boundingRect(contours[0])[2]
        #box_height = cv.boundingRect(contours[0])[3]
        #print("remaining contour area:", box_width*box_height)
        #print(type(contours)) #debug only
        #print(contours) #debug only
        #print("number of contours:", len(contours))
        #print(len(hierarchy))



        if len(contours) == 0:
            status = "no valid contour found"
            left_contour = 0
            right_contour = 0

        else: #always pick first contour, should be the biggest one (sometimes we detect almost the same contour multiple times)
        #print("handling one contour case")
            status = "Ok"
            contour_direction = hierarchy[0][3]
            #print("hieararchy", contour_direction)
            single_contour = contours[0]
            #contour direction => -1 = counter-clockwise, != -1 => clockwise
            if contour_direction != -1: #if order of coordiante points is not counter clockwise, we turn it around
                single_contour = single_contour[::-1]
                #print("flipped direction")

            #print(single_contour)
            #print(single_contour[len(single_contour)//2])
            x_start = single_contour[0,0,0] 
            #check if our contour start point is on left or right side of nozzle, changes depending on how high the contour goes on each side (higher side is start side)
            if (ll_limit<x_start<lr_limit): 
                start_side = 0
                #print("start left")
                #get list of indices where our x value is in the range accepted around the right nozzle edge
                id_other_side_x = np.where(np.logical_and(single_contour[:,0,0]>rl_limit, single_contour[:,0,0]<rr_limit))[0]
                #search in that indices range for the lowest y_value => highest point
                y_max_other_side = np.where(single_contour[id_other_side_x[0]:id_other_side_x[-1],0,1] == np.min(single_contour[id_other_side_x[0]:id_other_side_x[-1],0,1]) )[0][0]
                #calculate the index of that point, by adding the offset created by the list 
                id_top_other_side = id_other_side_x[0]+y_max_other_side
            else:
                start_side = 1
            # print("start right")
                #since we are on right side, we have to flip it around to go clock wise, thus along lower side of contour 
                single_contour = single_contour[::-1] 
                #get list of indices where our x value is in the range accepted around the left nozzle edge
                id_other_side_x = np.where(np.logical_and(single_contour[:,0,0]>ll_limit, single_contour[:,0,0]<lr_limit))[0]
                #search in that indices range for the lowest y_value => highest point
                y_max_other_side = np.where(single_contour[id_other_side_x[0]:id_other_side_x[-1],0,1] == np.min(single_contour[id_other_side_x[0]:id_other_side_x[-1],0,1]) )[0][0]
                #calculate the index of that point, by adding the offset created by the list 
                id_top_other_side = id_other_side_x[0]+y_max_other_side
                    
            single_contour = single_contour[:id_top_other_side] #take only part of contour until top of other side, after that the contour wraps around/runs back on inside of edge, we dont need that part
            
            y_max = np.max(single_contour[:,0,1]) #get max y_value
            idx_max_y = np.where(single_contour[:,0,1] == y_max)[0] #get list of index where our y is equal y_max

            #cv.line(adgc, (0, y_max), (img_width, y_max), (255,0,0), 1, cv.LINE_AA) #debug only
            #tv(adgc, size = 0.5)
            
            #print("idx_max_y", idx_max_y)
            if len(idx_max_y) == 1:
                idx_max_y_left = idx_max_y[0]
                idx_max_y_right = idx_max_y[0]
            else:
                if start_side == 0: # in this case we walk around counter clock wise, so first entry we find in list belongs to left side
                    idx_max_y_left = idx_max_y[0]
                    idx_max_y_right = idx_max_y[-1]
                else: # in this case we walk around clock wise, so first entry we find in list belongs to right side
                    idx_max_y_left = idx_max_y[-1]
                    idx_max_y_right = idx_max_y[0]
            

            #check if we run into image edge, if so, don't split contour according to y_max, but along edge
            idx_imgedge_x = np.where(single_contour[:,0,0] == imgedge_pixel_dist)[0] #get list of index where our x is equal 0, left image edge
            if len(idx_imgedge_x) == 0: 
                idx_imgedge_x = np.where(single_contour[:,0,0] == img_width-imgedge_pixel_dist)[0] #get list of index where our x is equal img_width, right image edge

            # #print(idx_imgedge_x)
            k = len(idx_imgedge_x)
            if k != 0:
                if start_side == 0:
                    idx_max_y_left = idx_imgedge_x[0]
                    idx_max_y_right = idx_imgedge_x[-1]
                else: # in this case we walk around clock wise, so first entry we find in list belongs to right side
                    idx_max_y_left = idx_imgedge_x[-1]
                    idx_max_y_right = idx_imgedge_x[0]

            #old
            #idx_max_y_mid = idx_max_y[len(idx_max_y)//2] #get index at middle of list , e.g. we could have more than one pixel at y_max, we want to break apart at middle of it
            #print("mid index", idx_max_y_mid, "of", len(idx_max_y))
            
            #if we start on left side, our coordinates are sorted counter clockwise
            if start_side == 0:
                left_contour = single_contour[:idx_max_y_left] #create left part of contour 
                right_contour = single_contour[idx_max_y_right:][::-1] #create right part of contour, flip its direction to have the normal start at nozzle orientation
            #if we start on right side, our coordinates are sorted clockwise
            else:
                left_contour = single_contour[idx_max_y_left:][::-1] #create left part of contour 
                right_contour = single_contour[:idx_max_y_right] #create right part of contour, flip its direction to have the normal start at nozzle orientation
                
        
            #Calculate m and b for lower nozzle edge  #points are [[xhigh,xlow],[yhigh,ylow]] 
            if (nozzlerightlowx-nozzleleftlowx) != 0:  
                m = (nozzlerightlowy-nozzleleftlowy)/(nozzlerightlowx-nozzleleftlowy) # m = (y1-y0)/(x1-x0)
                b = nozzlerightlowy-m*nozzlerightlowx #b = yi-m*xi

            #we just plug in the edge points into our line definition and check if it is solution, if so we found the intersection point
            u_idx_left = 0
            for c in left_contour[:,0]:
                if c[1] == round(m*c[0] +b): #round() slighly faster than int(np.rint)
                    break 
                u_idx_left  += 1 

            u_idx_right = 0
            for c in right_contour[:,0]:
                if c[1] == round(m*c[0] +b):
                    break 
                u_idx_right  += 1
            
            left_contour = left_contour[u_idx_left:]
            right_contour = right_contour[u_idx_right:]

            
            #Now lets try to cut the end of the jet in a perpendicular manner, 
            # get the normal for left and right contour.
            #approx short end section of contour with a line.

            #first approx the left/right contour with a bspline to make it a bit smoother
            #number of points to create out of bspline, this will be the moothed contour
            left_bspline_points = int(len(left_contour[:,0,1])/10) #1/10 of original number of points seems to work fine
            right_bspline_points = int(len(right_contour[:,0,1])/10)
            #array shaping should be done better
            left_contour_l = (left_contour[:,0,1], left_contour[:,0,0]) #(y, x)
            o = []
            for i in range(len(left_contour_l[0])):
                c = [left_contour_l[1][i],left_contour_l[0][i]]
                o.append(c)
            
            left_contour_smoothed = scipy_bspline(o,left_bspline_points,degree=5) #degree 5 seems to work well
            x_smoothed_left = left_contour_smoothed[:,0].astype('int')
            y_smoothed_left = left_contour_smoothed[:,1].astype('int')

            right_contour_l = (right_contour[:,0,1], right_contour[:,0,0]) #(y, x)
            o = []
            for i in range(len(right_contour_l[0])):
                c = [right_contour_l[1][i],right_contour_l[0][i]]
                o.append(c)

            right_contour_smoothed = scipy_bspline(o,right_bspline_points,degree=5)
            x_smoothed_right = right_contour_smoothed[:,0].astype('int')
            y_smoothed_right = right_contour_smoothed[:,1].astype('int')

            #lower end points of our smoothed contours, should be equal the original contour (bspline runs through end points)
            x1_left = x_smoothed_left[-1] #end point x, left contour
            y1_left = y_smoothed_left[-1] #end point y, right contour
            x1_right = x_smoothed_right[-1] #end point x, right contour
            y1_right = y_smoothed_right[-1] #end point y, right contour

            # how many of our smoothed points to use to approximate slope of end section
            approx_dist = 5 #five seems to work fine 
            #get the slope and offset of the end section and of the normal 
            ml, bl, mnl, bnl = getlinereg(x_smoothed_left[-approx_dist:][::-1], y_smoothed_left[-approx_dist:][::-1])
            mr, br, mnr, bnr = getlinereg(x_smoothed_right[-approx_dist:][::-1], y_smoothed_right[-approx_dist:][::-1])
            
            #now try to find intersection of the normals with the opposite edge
            len_right_contour = len(right_contour)
            idx_right = len_right_contour-1
            #print(idx_right)
            if mnl == np.inf: #if we have a vertical normal line 
                for c in right_contour[:,0][::-1]: #goes backwards (starting at end point) through contour coordinates, c[1] = y, c[0] = x
                    if c[0] == x1_left: #check where our right coordinate has same x_value as left contour end point -> on vertical normal line
                        break 
                    idx_right  -= 1
                #cv.line(adgc, (x1_left, 0), (x1_left, img_height), (0,255,255), 1, cv.LINE_AA)
            elif mnl == 0: #if we have a horizontal normal line
                for c in right_contour[:,0][::-1]: #goes backwards (starting at end point) through contour coordinates, c[1] = y, c[0] = x
                    if c[1] == y1_left: #check where our right coordinate has same y_value as left contour end point -> on horizontal normal line
                        break 
                    idx_right  -= 1
                #cv.line(adgc, (0, y1_left), (img_width, y1_left), (0,255,255), 1, cv.LINE_AA)
            else: #if we don't have vertical nor horizontal line
                for c in right_contour[:,0][::-1]: #goes backwards (starting at end point) through contour coordinates, c[1] = y, c[0] = x
                    yc = round(mnl*c[0] +bnl) #plug into line equation, if it solves it it is on line 
                    if abs(yc-c[1]) <=5: #sometimes we have some rounding/ staircase error -> check if we are within reasonable range
                        break 
                    idx_right  -= 1 
                if idx_right < len_right_contour/2: #to catch unplausible cut, e.g. sharp drop form, normal bad approximated
                    idx_right = -1 


            #     #debug only
            #     y1line = round(bnl) 
            #     y2line = round(mnl*img_width+bnl)
            #     cv.line(adgc, (0, y1line), (img_width, y2line), (0,255,255), 1, cv.LINE_AA)
            
            # x_cross = right_contour[:,0,0][idx_right] 
            # y_cross = right_contour[:,0,1][idx_right]

            # cv.drawMarker(adgc, (x1_left, y1_left), color=(0,0,255), markerType = cv.MARKER_CROSS, markerSize=15, thickness=3)
            # cv.drawMarker(adgc, (x_cross, y_cross), color=(255,0,0), markerType = cv.MARKER_CROSS, markerSize=15, thickness=3)
            # cv.drawMarker(adgc, (x1_right, y1_right), color=(0,255,255), markerType = cv.MARKER_CROSS, markerSize=15, thickness=3)

            len_left_contour = len(left_contour)
            idx_left = len_left_contour-1
            #print(idx_left)
            if mnr == np.inf: #if we have a vertical normal line 
                for c in left_contour[:,0][::-1]: #goes backwards (starting at end point) through contour coordinates, c[1] = y, c[0] = x
                    if c[0] == x1_right: #check where our right coordinate has same x_value as left contour end point -> on vertical normal line
                        break 
                    idx_left  -= 1
                #cv.line(adgc, (x1_right, 0), (x1_right, img_height), (0,255,0), 1, cv.LINE_AA)
            elif mnr == 0: #if we have a horizontal normal line
                for c in left_contour[:,0][::-1]: #goes backwards (starting at end point) through contour coordinates, c[1] = y, c[0] = x
                    if c[1] == y1_right: #check where our right coordinate has same y_value as left contour end point -> on horizontal normal line
                        break 
                    idx_left  -= 1
                #cv.line(adgc, (0, y1_right), (img_width, y1_right), (0,255,0), 1, cv.LINE_AA)
            else: #if we don't have vertical nor horizontal line
                for c in left_contour[:,0][::-1]: #goes backwards (starting at end point) through contour coordinates, c[1] = y, c[0] = x
                    yc = round(mnr*c[0] +bnr)
                    if abs(yc-c[1]) <=5 : #sometimes we have some rounding/ staircase error -> check if we are within reasonable range
                        break
                    idx_left  -= 1
                if idx_left < len_left_contour/2: #to catch unplausible cut, e.g. sharp drop form, normal bad approximated
                    idx_left = -1

            #     #debug only
            #     y1line = round(bnr)
            #     y2line = round(mnr*img_width+bnr)
            #     cv.line(adgc, (0, y1line), (img_width, y2line), (0,255,0), 1, cv.LINE_AA)

            # x_cross = left_contour[:,0,0][idx_left]
            # y_cross = left_contour[:,0,1][idx_left]

            # cv.drawMarker(adgc, (x1_right, y1_right), color=(0,0,255), markerType = cv.MARKER_STAR, markerSize=15, thickness=3)
            # cv.drawMarker(adgc, (x_cross, y_cross), color=(255,0,0), markerType = cv.MARKER_STAR, markerSize=15, thickness=3)
            # cv.drawMarker(adgc, (x1_left, y1_left), color=(0,255,255), markerType = cv.MARKER_STAR, markerSize=15, thickness=3)


            #tv(adgc, size=0.5, time=1000)
            
            #cut the contours 
            left_contour = left_contour[:idx_left]
            right_contour = right_contour[:idx_right]

            # lets do a final smoothing run on the final cutted contours 
            left_bspline_points = int(len(left_contour[:,0,1])) #will give us the same number of points as the raw contours
            right_bspline_points = int(len(right_contour[:,0,1]))

            #array shaping should be done better
            left_contour_l1 = left_contour[:,0,1][0::10]
            left_contour_l0 = left_contour[:,0,0][0::10]
            #make sure last point is in shortened contour list, sometimes not the case as number of points not always dividable by 10 withour rest 
            if left_contour_l1[-1] != left_contour[:,0,1][-1] or left_contour_l0[-1] != left_contour[:,0,0][-1]:
                left_contour_l1 = np.append(left_contour_l1, left_contour[:,0,1][-1])
                left_contour_l0= np.append(left_contour_l0, left_contour[:,0,0][-1])

            left_contour_l = (left_contour_l1, left_contour_l0) #(y, x)
            
            o = []
            for i in range(len(left_contour_l[0])):
                c = [left_contour_l[1][i],left_contour_l[0][i]]
                o.append(c)
            
            left_contour_smoothed = scipy_bspline(o,left_bspline_points,degree=5) #degree 5 seems to work well
            x_smoothed_left = left_contour_smoothed[:,0].astype('int')
            y_smoothed_left = left_contour_smoothed[:,1].astype('int')


            right_contour_l1 = right_contour[:,0,1][0::10]
            right_contour_l0 = right_contour[:,0,0][0::10]
            #make sure last point is in shortened contour list, sometimes not the case as number of points not always dividable by 10 withour rest 
            if right_contour_l1[-1] != right_contour[:,0,1][-1] or right_contour_l0[-1] != right_contour[:,0,0][-1]:
                right_contour_l1 = np.append(right_contour_l1, right_contour[:,0,1][-1])
                right_contour_l0= np.append(right_contour_l0, right_contour[:,0,0][-1])

            right_contour_l = (right_contour_l1, right_contour_l0) #(y, x)

            o = []
            for i in range(len(right_contour_l[0])):
                c = [right_contour_l[1][i],right_contour_l[0][i]]
                o.append(c)

            right_contour_smoothed = scipy_bspline(o,right_bspline_points,degree=5)
            x_smoothed_right = right_contour_smoothed[:,0].astype('int')
            y_smoothed_right = right_contour_smoothed[:,1].astype('int')


            #lets bring our contour data in shape
            left_contour = (left_contour[:,0,1], left_contour[:,0,0]) #(y, x)
            right_contour = (right_contour[:,0,1], right_contour[:,0,0])
            left_contour_smoothed = (y_smoothed_left, x_smoothed_left)
            right_contour_smoothed = (y_smoothed_right, x_smoothed_right)
    except:
        adgimage = image
        status = "Contours not found - exception occured"
        left_contour_smoothed = 0
        right_contour_smoothed = 0
        left_contour = 0
        right_contour = 0
        ll_limit = 0
        lr_limit = 0
        rl_limit = 0
        rr_limit = 0
        all_contours = 0
        contours = 0
        y_min_l = 0
        y_min_r = 0


    return adgimage, status, left_contour_smoothed, right_contour_smoothed, left_contour, right_contour, ll_limit, lr_limit, rl_limit, rr_limit, all_contours, contours, y_min_l, y_min_r



#get midline from jet contours
def getMidlinefromContours(left_contour, right_contour):
    try:
        #get out contour top/bottom ends 
        left_contour_top_end = (left_contour[1][0], left_contour[0][0]) # (x, y)
        left_contour_bottom_end = (left_contour[1][-1], left_contour[0][-1])
        right_contour_top_end = (right_contour[1][0], right_contour[0][0])
        right_contour_bottom_end = (right_contour[1][-1], right_contour[0][-1])
        #calculate points (x, y) for top/bottom triangle 
        A = getTrianglePoint(right_contour_bottom_end, left_contour_bottom_end, 20) 
        A2 = getTrianglePoint(left_contour_top_end, right_contour_top_end, 10)
        #get max x, y values to determine the ROI we have to look at 
        x_max = max(max(left_contour[1]), max(right_contour[1]), A[0], A2[0])
        x_min = min(min(left_contour[1]), min(right_contour[1]), A[0], A2[0])
        y_max = max(max(left_contour[0]), max(right_contour[0]), A[1], A2[1])
        y_min = min(min(left_contour[0]), min(right_contour[0]), A[1], A2[1])
        #add a little offset at the image edges to avoid issues with cv operations on edge
        edge_width = 10
        dx = x_max-x_min + 2*edge_width
        dy = y_max-y_min + 2*edge_width
        #create our working image
        jet_img = np.zeros((dy,dx), dtype="uint8")
        #offset our coordiantes 
        left_contour = (left_contour[0]-(y_min-edge_width), left_contour[1]-(x_min-edge_width))
        right_contour = (right_contour[0]-(y_min-edge_width), right_contour[1]-(x_min-edge_width))
        left_contour_top_end = (left_contour[1][0], left_contour[0][0]) # (x, y)
        left_contour_bottom_end = (left_contour[1][-1], left_contour[0][-1])
        right_contour_top_end = (right_contour[1][0], right_contour[0][0])
        right_contour_bottom_end = (right_contour[1][-1], right_contour[0][-1])
        A = (A[0]-(x_min-edge_width), A[1]-(y_min-edge_width))
        A2 = (A2[0]-(x_min-edge_width), A2[1]-(y_min-edge_width))
        #use draw line function to make sure there are no gaps left between any two coordinate points
        #especially important for smoothed/interpolated coordinates
        cva.lineBetweenPoints(jet_img, left_contour, dimx=1, dimy=0, thickness=1, color=255, linetype=cv.LINE_4)
        cva.lineBetweenPoints(jet_img, right_contour, dimx=1, dimy=0, thickness=1, color=255, linetype=cv.LINE_4)
        #close edges 
        cv.line(jet_img, left_contour_top_end,right_contour_top_end, 255, thickness=1 ,lineType=cv.LINE_4) 
        cv.line(jet_img, left_contour_bottom_end,right_contour_bottom_end, 255, thickness=1 ,lineType=cv.LINE_4)
        #floodfill and invert
        cv.floodFill(jet_img, None , (0,0), 255)
        jet_img = cv.bitwise_not(jet_img)
        filled_jet_img = jet_img.copy()
        #draw in rectangle extensions from points A, A2
        points = np.array([right_contour_bottom_end,left_contour_bottom_end,A]) #format point array
        cv.fillPoly(jet_img, [points], color=255)
        points = np.array([left_contour_top_end, right_contour_top_end, A2])
        cv.fillPoly(jet_img, [points], color=255)
        #make sure no small gaps are left 
        kernel = np.ones((3,3), np.uint8)
        jet_img = cv.morphologyEx(jet_img, cv.MORPH_CLOSE, kernel)
        #get the skeleton
        middle_line_img = skimage.morphology.skeletonize(jet_img, method="zhang") #use skimage skeleton, approx 15-30ms, is 10x faster than opencv
        middle_line_img = middle_line_img.astype("uint8")*255 #returns bool array, convert back to uint array    
        #now crop our too long skeleton with our filled jet ROI 
        middle_line_img = cv.bitwise_and(middle_line_img, filled_jet_img)
        #get middle line pixel coordinates
        np_middle_line_coordinates =  np.nonzero(middle_line_img)
        #get last point on top and bottom 
        middle_line_bottom = getLastPointBottom(np_middle_line_coordinates) #returns point (x,y)
        middle_line_top = getLastPointTop(np_middle_line_coordinates)
        #calculate our "ideal" midline start and end points -> in between left/right contour top/bottom end
        x_top_mid = round((left_contour[1][0]+right_contour[1][0])/2)
        y_top_mid = round((left_contour[0][0]+right_contour[0][0])/2)
        x_bottom_mid = round((left_contour[1][-1]+right_contour[1][-1])/2)
        y_bottom_mid = round(( left_contour[0][-1]+right_contour[0][-1])/2)

        #top line slope and offset,  getlinetp(x1, y1, x2, y2), left_contour_top_end = (left_contour[1][0], left_contour[0][0]) # (x, y)
        m, b, _, _ = getlinetp(left_contour_top_end[0], left_contour_top_end[1], right_contour_top_end[0], right_contour_top_end[1])
        #search for intersection of middle line pixel coordinates with top line
        intersect_top_idc = []
        if m == np.inf: #if we have a vertical normal line 
        # print("vertical line")
            for i in range(len(np_middle_line_coordinates[0])): #goes backwards (starting at end point) through contour coordinates, c[1] = y, c[0] = x
                if np_middle_line_coordinates[1][i] == left_contour_top_end[0]: #check where our right coordinate has same x_value as left contour end point -> on vertical normal line
                    intersect_top_idc.append(i) 
        elif m == 0: #if we have a horizontal normal line
            #print("horizontal line")
            for i in range(len(np_middle_line_coordinates[0])): #goes backwards (starting at end point) through contour coordinates, c[1] = y, c[0] = x
                if np_middle_line_coordinates[0][i] == left_contour_top_end[1]: #check where our right coordinate has same y_value as left contour end point -> on horizontal normal line
                    intersect_top_idc.append(i) 
        else: #if we don't have vertical nor horizontal line
            #print("mb line")
            for i in range(len(np_middle_line_coordinates[0])): #goes backwards (starting at end point) through contour coordinates, c[1] = y, c[0] = x
                yc = m*np_middle_line_coordinates[1][i] +b #plug into line equation, if it solves it it is on line 
                if abs(yc-np_middle_line_coordinates[0][i]) <=2: #sometimes we have some rounding/ staircase error -> check if we are within reasonable range
                    intersect_top_idc.append(i) 

        #print("length top intersect:", len(intersect_top_idc))
        #now look at the intersection points and calculate the distance to the ideal top-mid point
        dist_top = np.inf
        for i in intersect_top_idc:
            x = np_middle_line_coordinates[1][i]
            y = np_middle_line_coordinates[0][i]
            d = np.sqrt((x-x_top_mid)**2 + (y-y_top_mid)**2)
            #print("distance top", d)
            if d < dist_top: #if we found a smaller distance
                #print("new_top_start")
                middle_line_top = (x, y) #make it our new middle_line_top point

        #do the same for the bottom line,  
        m, b, _, _ = getlinetp(left_contour_bottom_end[0], left_contour_bottom_end[1], right_contour_bottom_end[0], right_contour_bottom_end[1])

        intersect_bottom_idc = []
        if m == np.inf: #if we have a vertical normal line 
            #print("vertical line")
            for i in range(len(np_middle_line_coordinates[0])): #goes backwards (starting at end point) through contour coordinates, c[1] = y, c[0] = x
                if np_middle_line_coordinates[1][i] == left_contour_top_end[0]: #check where our right coordinate has same x_value as left contour end point -> on vertical normal line
                    intersect_bottom_idc.append(i) 
        elif m == 0: #if we have a horizontal normal line
            #print("horizontal line")
            for i in range(len(np_middle_line_coordinates[0])): #goes backwards (starting at end point) through contour coordinates, c[1] = y, c[0] = x
                if np_middle_line_coordinates[0][i] == left_contour_top_end[1]: #check where our right coordinate has same y_value as left contour end point -> on horizontal normal line
                    intersect_bottom_idc.append(i) 
        else: #if we don't have vertical nor horizontal line
            #print("mb line")
            for i in range(len(np_middle_line_coordinates[0])): #goes backwards (starting at end point) through contour coordinates, c[1] = y, c[0] = x
                yc = m*np_middle_line_coordinates[1][i] +b #plug into line equation, if it solves it it is on line 
                if abs(yc-np_middle_line_coordinates[0][i]) <=2: #sometimes we have some rounding/ staircase error -> check if we are within reasonable range
                    intersect_bottom_idc.append(i) 
        #print("length bottom intersect:", len(intersect_bottom_idc))

        #now look at the intersection points and calculate the distance to the ideal bottom-mid point
        dist_bottom = np.inf
        middle_line_bottom
        for i in intersect_bottom_idc:
            x = np_middle_line_coordinates[1][i]
            y = np_middle_line_coordinates[0][i]
            d = np.sqrt((x-x_bottom_mid)**2 + (y-y_bottom_mid)**2)
            #print("bottom distance:", d)
            if d < dist_bottom:
                #print("new bottom_start")
                middle_line_bottom = (x, y)
        
        #now create a nx graph for our middle line pixels 
        graph = sknw.build_sknw(middle_line_img, multi=False, iso=False, ring=False, full=True)
        #get a list of our nodes 
        nodes = list(graph.nodes()) #get a list of the nodes -> number them 0, 1, ...
        # set default start/stop node
        start_node = nodes[0] 
        stop_node = nodes[-1]
        #now go trhough the nodes and see which one is on our top/bottom middle line -> reject branches at start/end of middle line
        for i in nodes: # go through the nodes
            node_centroid = graph.nodes[i]["o"] # (y, x) #get their centroid coordinates 
            #print(node_centroid)
            if node_centroid[0] == middle_line_top[1] and node_centroid[1] == middle_line_top[0]:  #see if they match with our middle_line_top/bottom coordiantes
                start_node = i # if so set this node as start/stop node 
                #print("new start node set")
            if node_centroid[0] == middle_line_bottom[1] and node_centroid[1] == middle_line_bottom[0]:
                stop_node = i
                #print("new stop node set")

        #now get all simple paths between the start and stop node
        # a simple path is a path from start to stop with no repeated nodes, so we eliminate the branches along the way
        #shortest_simple_paths returns simple paths between start and stop sorted by length
        shortest_path = list(nx.shortest_simple_paths(graph, start_node, stop_node))[0]
        #get the new midline pixel coordinate set from the simple path 
        x_coordinates = np.array([])
        y_coordinates = np.array([])
        for i in range(len(shortest_path)-1):
            coordinates = graph.edges[shortest_path[i], shortest_path[i+1]]["pts"] # [[y,x],[y,x],...] , each edge contains the corresponding midline pixel coordinates 
            x_coordinates = np.append(x_coordinates, coordinates[:,1]) 
            y_coordinates = np.append(y_coordinates, coordinates[:,0])
            #print(coordinates)
            #print(x_coordinates)
            #print(y_coordinates)
        x_coordinates = x_coordinates.astype("int")
        y_coordinates = y_coordinates.astype("int")

        #now smooth midline a bit, same like contours 
        bspline_points = int(len(x_coordinates)) #will give us the same number of points as the raw midline

        #array shaping should be done better
        g_midline_x = x_coordinates[0::10]
        g_midline_y = y_coordinates[0::10]
        #make sure last point is in shortened contour list, sometimes not the case as number of points not always dividable by 10 withour rest 
        if g_midline_x[-1] != x_coordinates[-1] or g_midline_y[-1] != y_coordinates[-1]:
            g_midline_x = np.append(g_midline_x, x_coordinates[-1])
            g_midline_y= np.append(g_midline_y, y_coordinates[-1])

        g_midline_l = (g_midline_y, g_midline_x) #(y, x)
        
        o = []
        for i in range(len(g_midline_l[0])):
            c = [g_midline_l[1][i],g_midline_l[0][i]]
            o.append(c)
        
        midline_smoothed = scipy_bspline(o,bspline_points,degree=5) #degree 5 seems to work well
        x_smoothed = midline_smoothed[:,0].astype('int')
        y_smoothed = midline_smoothed[:,1].astype('int')

        #Finally, we have to ofset our midline coordinates, as we cropped/translated our image in the beginning
        x_coordinates = x_coordinates + (x_min-edge_width)
        y_coordinates = y_coordinates + (y_min-edge_width)
        g_middle_line_coordinates = (y_coordinates, x_coordinates) # ([y, y, ...], [x, x, ...])
        x_smoothed = x_smoothed + (x_min-edge_width)
        y_smoothed = y_smoothed + (y_min-edge_width)
        middle_line_coordinates_smoothed = (y_smoothed, x_smoothed) # ([y, y, ...], [x, x, ...])

        # #debug only
        # # # draw image
        # plt.imshow(img, cmap='gray')
        # # draw edges by pts
        # for (s,e) in graph.edges():
        #     ps = graph[s][e]['pts']
        #     #print(ps)
        #     plt.plot(ps[:,1], ps[:,0], 'o','green')
        # # draw node by o
        # nodes = graph.nodes()
        # ps = np.array([nodes[i]['o'] for i in nodes])
        # plt.plot(ps[:,1], ps[:,0], 'r.')
        # # title and show
        # plt.title('Build Graph')
        # plt.show()
            
        # OLD, ISSUE: only works for horizontal search
        #search for groups of y-coordinate values
        # i = 0
        # lenl = []
        # idxl =[]
        # for k, g in groupby(middle_line_coordinates[0]):
        #     l = len(list(g))
        #     lenl.append(l)
        #     idxl.append(i)
        #     i += l 
        # go through those groups 
        # for i in range(len(idxl)):
        #     x = middle_line_coordinates[1][idxl[i]:idxl[i]+lenl[i]] #get the list of x-coordinates with this y-value
        #     xdiff = np.diff(x) #get the difference between them 
        #     if len(xdiff) != 0: 
        #         maxdiff = max(xdiff) #get the maximum of the difference 
        #         if maxdiff > 1: #if it is bigger one, there is a gap between the pixels -> could be a branching section
        #             #calculate the distance of the pixels in that group to the top/bottom mid point 

        #             print("maxdiff greater 1")
        #             print(x)
        #             print(middle_line_coordinates[0][idxl[i]])

        status = "Ok"
        return g_middle_line_coordinates, middle_line_coordinates_smoothed, status
    except:
        g_middle_line_coordinates = 0
        middle_line_coordinates_smoothed = 0
        status = "Midline not found - exception occured"

        return g_middle_line_coordinates, middle_line_coordinates_smoothed, status 

#get jet vectors 
def getJetVector(nozzle_mid_point, line_coordinates, vector_positions, pixel_size = 1):
    #nozzle_x = nozzle_mid_point[0][0]
    #nozzle_y = nozzle_mid_point[1][0]
    try:

        vectors = []
        vector_indicies = []
        #jet_angles = np.zeros(len(angle_positions))
        #angle_indicies = np.zeros(len(angle_positions))
        y_d =(np.max(line_coordinates[0]) - nozzle_mid_point[1][0]) #get y_distance of midline in pixels
        for i in range(len(vector_positions)):
            #print("angle position:", angle_positions[i])
            y_dist = np.rint((vector_positions[i]/100) * y_d) #get pixel count corresponding to relative (%) vector position along y-distance spanned by midline
            
            #print("y_dist:", y_dist) #debug
            idx = np.where(line_coordinates[0] == (y_dist + nozzle_mid_point[1][0])) #check index of the y-pos in our mid-line coordinates list
            #print("idx:", idx)
            if len(idx[0]) == 0: #if this position is not in our list
                vectors.append([-1,-1]) #append if invalid y-pos, 
                vector_indicies.append(-1)
                continue #skip rest of current for-loop
            elif len(idx[0]) > 1: #more than one entry, can happen if mid-line runs horizontal for a short distance => staircase
                id = int(np.rint(np.mean(idx[0]))) #take middle point of horizontal section, round to next integer
                #print("multi entries, id:", id)
            else: #if only one entry in list, take that one 
                id = idx[0][0]

            vx = (line_coordinates[1][id] - nozzle_mid_point[0][0]) * pixel_size #calculate vector components, follows CV coordinate system, X to right, Y down
            vy = (line_coordinates[0][id] - nozzle_mid_point[1][0]) * pixel_size                                                                      
            vectors.append([vx,vy]) #append vector                                                                                      
            vector_indicies.append(id) #save the index of the mid_line point we used 


        status = "Ok"
    except:
        status = "ERROR - Exception occured while caclulating jet vectors."
        #print("ERROR - Exception occured while caclulating jet angles.")
        vectors = [[-1,-1]]
        vector_indicies = [-1]

    return vectors, vector_indicies, status


#get jet lag - USE THIS ONE
def getJetLag(nozzle_mid_point, mid_line_coordinates, mid_line_indices, pixel_size=1):
    #nozzle_x = nozzle_mid_point[0][0]
    #nozzle_y = nozzle_mid_point[1][0]
    try:

        jet_lags = []
        #jet_angles = np.zeros(len(angle_positions))
        #angle_indicies = np.zeros(len(angle_positions))
        for i in mid_line_indices:
            #print("angle position:", angle_positions[i])
            if i == -1:
                jet_lags.append(np.NAN) #append Nan if invalid midline index
            else:
                jet_lags.append((nozzle_mid_point[0][0]-mid_line_coordinates[1][i])*pixel_size) #calculate angle 
            #print("jet angles:", jet_angles)
        status = "Ok"
    except:
        status = "ERROR - Exception occured while caclulating jet lag distances."
        #print("ERROR - Exception occured while caclulating jet angles.")
        jet_lags = [np.NAN]

    return jet_lags, status

#get jet angles
def getJetAngleOLD(nozzle_mid_point, ref_distance, mid_line_coordinates, angle_positions, relative=True, pixel_size = 1):
    #nozzle_x = nozzle_mid_point[0][0]
    #nozzle_y = nozzle_mid_point[1][0]
    try:

        jet_angles = []
        angle_indicies = []
        #jet_angles = np.zeros(len(angle_positions))
        #angle_indicies = np.zeros(len(angle_positions))
        for i in range(len(angle_positions)):
            #print("angle position:", angle_positions[i])
            if relative == True:
                y_dist= np.rint((angle_positions[i]/100) * ref_distance) #caclulate abs position along midline 
            else:
                y_dist = np.rint(angle_positions[i]/pixel_size) #if we give absolute distance position in um 
            
            #print("y_dist:", y_dist) #debug
            idx = np.where(mid_line_coordinates[0] == (y_dist + nozzle_mid_point[1][0])) #check index of the y-pos in our mid-line coordinates list
            #print("idx:", idx)
            if len(idx[0]) == 0: #if this position is not in our list
                jet_angles.append(-180) #append Nan if invalid y-pos, e.g. too low percentage 
                angle_indicies.append(-1)
                continue #skip rest of current for-loop
            elif len(idx[0]) > 1: #more than one entry, can happen if mid-line runs horizontal for a short distance => staircase
                id = int(np.rint(np.mean(idx[0]))) #take middle point of horizontal section, round to next integer
                #print("multi entries, id:", id)
            else: #if only one entry in list, take that one 
                id = idx[0][0]
            #print("mid line x:", mid_line_coordinates[1][id] )
            #print("delta x:", nozzle_mid_point[1]-mid_line_coordinates[1][id])
            jet_angles.append((np.arctan2((nozzle_mid_point[0][0]-mid_line_coordinates[1][id]),y_dist)) * 180/np.pi) #calculate angle 
            angle_indicies.append(id) #save the index of the mid_line point we used 
            #print("jet angles:", jet_angles)
        last_angle = np.arctan2((nozzle_mid_point[0][0]-mid_line_coordinates[1][-1]),ref_distance) * 180/np.pi
        status = "Ok"
    except:
        status = "ERROR - Exception occured while caclulating jet angles."
        #print("ERROR - Exception occured while caclulating jet angles.")
        jet_angles = [-180]
        last_angle = -180
        angle_indicies = [-1]

    return jet_angles, last_angle, angle_indicies, status
#get jet angles - USE THIS ONE
def getJetAngle(nozzle_mid_point, mid_line_coordinates, mid_line_indices):
    #nozzle_x = nozzle_mid_point[0][0]
    #nozzle_y = nozzle_mid_point[1][0]
    try:

        jet_angles = []
        #jet_angles = np.zeros(len(angle_positions))
        #angle_indicies = np.zeros(len(angle_positions))
        for i in mid_line_indices:
            #print("angle position:", angle_positions[i])
            if i == -1:
                jet_angles.append(-180) #append Nan if invalid midline index
            else:
                jet_angles.append((np.arctan2((nozzle_mid_point[0][0]-mid_line_coordinates[1][i]),(mid_line_coordinates[0][i]-nozzle_mid_point[1][0]))) * 180/np.pi) #calculate angle 
            #print("jet angles:", jet_angles)
        status = "Ok"
    except:
        status = "ERROR - Exception occured while caclulating jet angles."
        #print("ERROR - Exception occured while caclulating jet angles.")
        jet_angles = [-180]

    return jet_angles, status

#get all lengths along midline
def getLengthAlongMidline(mid_line_coordinates, pixel_size=1):
    try:
        all_midline_lengths = np.zeros(len(mid_line_coordinates[0])-1) #minus one as we have one length less than points 

        dist_sum = 0 
        for i in range(len(all_midline_lengths)):
            dist_sum += pixel_size * np.sqrt(np.square(mid_line_coordinates[0][i]-mid_line_coordinates[0][i+1])+np.square(mid_line_coordinates[1][i]-mid_line_coordinates[1][i+1])) #caclulate distance from one midline point to the next, sum up
            all_midline_lengths[i] = dist_sum #write into our return array, the length up to this point
        status = "Ok"

    except:
        all_midline_lengths = [-1]
        status = "Error finding all spinline lengths"
    
    return all_midline_lengths, status

#get jet lengths
def getJetLengthOLD(nozzle_mid_point, ref_distance, mid_line_coordinates, length_positions, pixel_size=1, relative=True):
    try:
        all_spinline_lengths = np.zeros(len(mid_line_coordinates[0])-1) #minus one as we have one length less than points 

        dist_sum = 0 
        for i in range(len(all_spinline_lengths)):
            dist_sum += pixel_size * np.sqrt(np.square(mid_line_coordinates[0][i]-mid_line_coordinates[0][i+1])+np.square(mid_line_coordinates[1][i]-mid_line_coordinates[1][i+1])) #caclulate distance from one midline point to the next, sum up
            all_spinline_lengths[i] = dist_sum #write into our return array, the length up to this point
        
            
        jet_lengths = []
        length_indices = [] 
        for i in range(len(length_positions)):
            if relative == True:
                y_dist = np.rint((length_positions[i]/100)*ref_distance) # y_distance in pixels from nozzle for requested angle position in % of deposition distance
            else:
                y_dist = np.rint(length_positions[i]/pixel_size) #if we give absolute distance position in pixels 

            idx = np.where(mid_line_coordinates[0] == (y_dist + nozzle_mid_point[1][0])) #check index of the y-pos in our mid-line coordinates list
            if len(idx[0]) == 0: #if this position is not in our list
                jet_lengths.append(-1) #append Nan if invalid y-pos, e.g. too low percentage, or too big/small absolute distance
                length_indices.append(-1)
                continue #skip rest of current for-loop
            elif len(idx[0]) > 1: #more than one entry, can happen if mid-line runs horizontal for a short distance => staircase
                id = int(np.rint(np.mean(idx[0]))) #take middle point of horizontal section, round to next integer
                #print("multi entries, id:", id)
            else: #if only one entry in list, take that one 
                id = idx[0][0]
                #print("id:", id)
            
            jet_lengths.append(all_spinline_lengths[id-1])
            length_indices.append(id)
        last_length = all_spinline_lengths[-1]
        status = "Ok"
    except:
        status = "ERROR - Exception occured while calculating jet length."
        jet_lengths = [-1]
        last_length = -1
        length_indices = [-1]

    return jet_lengths, last_length, length_indices, status

#get jet area
def getSpinlineArea2_skimage_V4OLD(nozzle_mid_point, ref_distance, midline_coordinates, left_edge_coordinates, right_edge_coordinates,area_positions, pixel_size=1, relative=True, approx_dist=25):
    try:
            #find minimum and maximum x/y edge coordinates 
        min_x = min( np.min(left_edge_coordinates[1]), np.min(right_edge_coordinates[1])) 
        max_x = max( np.max(left_edge_coordinates[1]), np.max(right_edge_coordinates[1]))
        min_y = min( np.min(left_edge_coordinates[0]), np.min(right_edge_coordinates[0]))
        max_y = max( np.max(left_edge_coordinates[0]), np.max(right_edge_coordinates[0]))

        #to have a little distance from filled jet
        min_x = max(0,min_x-5)
        max_x = max_x+5 
        min_y = max(0,min_y-5)
        max_y = max_y+5 

        jet_areas = [] #list of areas for given area positions
        area_edge_indices = []
        last_id = 0
        area_sum = 0
        surface_scaler = np.square(pixel_size)
        for i in range(len(area_positions)):
            print(i)
            #print("new diameter, i:", i) #debug only
            if relative == True:
                y_dist = np.rint((area_positions[i]/100)*ref_distance) # y_distance in pixels from nozzle for requested angle position in % of deposition distance
            else:
                y_dist = np.rint(area_positions[i]/pixel_size) #if we give absolute distance position in pixels 

            idx = np.where(midline_coordinates[0] == (y_dist + nozzle_mid_point[1][0])) #check index of the y-pos in our mid-line coordinates list
            if len(idx[0]) == 0: #if this position is not in our list
                jet_areas.append(-1) #append if invalid y-pos, e.g. too low percentage, or too big/small absolute distance
                area_edge_indices.append([-1,-1])
                continue #skip rest of current for-loop
            elif len(idx[0]) > 1: #more than one entry, can happen if mid-line runs horizontal for a short distance => staircase
                id = int(np.rint(np.mean(idx[0]))) #take middle point of horizontal section, round to next integer
                #print("multi entries, id:", id)
            else: #if only one entry in list, take that one 
                id = idx[0][0]


            #take points before and after our mid-line point, and calculate slope
            if id < approx_dist: # if we are very close to our nozzle, we can't go further back
                id_b = 0
            else:
                id_b = id-approx_dist

            if (id+approx_dist) > len(midline_coordinates[0])-1 : # if we are very close to the end of our mid line, we can't go further 
                id_f = -1
            else:
                id_f = id+approx_dist+1 #+1 because of stupid slice indexing

            x = midline_coordinates[1][id_b:id_f] #+1 because of stupid slice indexing
            y = midline_coordinates[0][id_b:id_f]

            j = id-id_b # i in x, y is our reference midpoint through which the normal should pass 
            #print(id, id_b, id_f, i)       
            _, _, m, b = getlineregI(x, y, j) #get the slope and intersect for the approximated normal through the given points 

            #now we have to find the intersection of our normal with the contours, we can do so by checking if our points on the edge statisfy the y=mx+b equation of our normal line
            left_edge_idx = -1 #init to -1 so we can check wheter we found an intersection
            right_edge_idx = -1 


            for j in range(len(left_edge_coordinates[0])):
                #print("left edge test:", left_edge_coordinates[0][i] - m*left_edge_coordinates[1][i] + b) #debug only
                if ((m*left_edge_coordinates[1][j] + b) - left_edge_coordinates[0][j])  < 1: #as we have discrete x/y values for our edges, sometime we dont find perfect match, thus check if we are close -> lines might intersect between pixels 
                    #print("found intersection with left edge, idx:", i)
                    left_edge_idx = j
                    break #we can stop our for loop 

            for j in range(len(right_edge_coordinates[0])):
                if ((m*right_edge_coordinates[1][j] + b) - right_edge_coordinates[0][j]) < 1:
                    #print("found intersection with right edge, idx:", i)
                    right_edge_idx = j
                    break #we can stop our for loop 
            
            area_edge_indices.append([left_edge_idx,right_edge_idx])
            
        
            if i == 0:
                #create closed jet image 
                cut_jet_img = np.zeros((max_y-min_y, max_x-min_x), dtype="uint8") 
                #draw part of the edge contour
                cut_jet_img[(left_edge_coordinates[0][:left_edge_idx+1]-min_y, left_edge_coordinates[1][:left_edge_idx+1]-min_x)] = 255
                cut_jet_img[(right_edge_coordinates[0][:right_edge_idx+1]-min_y, right_edge_coordinates[1][:right_edge_idx+1]-min_x)] = 255
                pix_count_left_edge = len(left_edge_coordinates[0][:left_edge_idx+1])
                pix_count_right_edge = len(right_edge_coordinates[0][:right_edge_idx+1])
                #calculate points of lower edge connection point, keep offset for cut image in mind
                p1x= left_edge_coordinates[1][left_edge_idx] - min_x
                p1y = left_edge_coordinates[0][left_edge_idx] - min_y
                p2x = right_edge_coordinates[1][right_edge_idx] - min_x
                p2y = right_edge_coordinates[0][right_edge_idx] - min_y
                #draw the upper and lower edge connection line
                a1, b1 = skimage.draw.line(left_edge_coordinates[0][0]-min_y, left_edge_coordinates[1][0]-min_x, right_edge_coordinates[0][0]-min_y, right_edge_coordinates[1][0]-min_x)
                cv.line(cut_jet_img, (p1x,p1y),(p2x,p2y), 255, thickness=1 ,lineType=cv.LINE_4) #we only count pixels of upper edge to current area, otherwise we would count it multiple times, so use faster cv.line function for lower edge as we don't need the pixels idx array returned
                pixel_count_upper = len(a1)-2 #-2 as we paint on top of edge pixels
                cut_jet_img[a1, b1] = 255
                #calculate the mid point of the area we try to fill 
                iy2 = int((midline_coordinates[0][0]+midline_coordinates[0][id])/2) - min_y
                ixa2 = np.where(cut_jet_img[iy2,:] == 255)
                ix2 = int((ixa2[0][0]+ixa2[0][-1])/2)
                #flood fill the area 
                fill_area = cv.floodFill(cut_jet_img, None , (ix2, iy2), 255)[0]  #might be too slow, lets see, could be done with findcontours too
                #tv(cut_jet_img_copy)
                #count the nonzero pixels => area and scale with pixel size
                area = ((pix_count_left_edge+pix_count_right_edge+pixel_count_upper+fill_area) * surface_scaler)
                area_sum += area
            # print("non zero area, pix count area:", np.count_nonzero(cut_jet_img_copy), (pix_count_left_edge+pix_count_right_edge+pixel_count_upper+pixel_count_lower+area))
                #area_sum += nonzero_count(cut_jet_img_copy.ravel()) * np.square(pixel_size) #slower :/
                #area_sum += (cut_jet_img_copy != 0).sum() * surface_scaler
                jet_areas.append(area) # convert to um2 and append to list
                p1x_last = p1x
                p1y_last = p1y
                p2x_last = p2x
                p2y_last = p2y
                last_id = id
                last_left_edge_idx = left_edge_idx
                last_right_edge_idx = right_edge_idx

            else:
                #create closed jet image 
                cut_jet_img = np.zeros((max_y-min_y, max_x-min_x), dtype="uint8") 
                #draw part of the edge contour for next section
                cut_jet_img[(left_edge_coordinates[0][last_left_edge_idx:left_edge_idx+1]-min_y ,left_edge_coordinates[1][last_left_edge_idx:left_edge_idx+1]-min_x)] = 255
                cut_jet_img[(right_edge_coordinates[0][last_right_edge_idx:right_edge_idx+1]-min_y,right_edge_coordinates[1][last_right_edge_idx:right_edge_idx+1]-min_x)] = 255
                pix_count_left_edge = len(left_edge_coordinates[0][last_left_edge_idx:left_edge_idx+1])
                pix_count_right_edge = len(right_edge_coordinates[0][last_right_edge_idx:right_edge_idx+1])
                #calculate points of lower edge connection point, keep offset for cut image in mind
                p1x = left_edge_coordinates[1][left_edge_idx] - min_x
                p1y = left_edge_coordinates[0][left_edge_idx] - min_y
                p2x = right_edge_coordinates[1][right_edge_idx] - min_x
                p2y = right_edge_coordinates[0][right_edge_idx] - min_y
                #draw the upper and lower edge connection line
                a1, b1 = skimage.draw.line(p1y_last, p1x_last, p2y_last, p2x_last)
                cv.line(cut_jet_img, (p1x,p1y),(p2x,p2y), 255, thickness=1 ,lineType=cv.LINE_4) #we only count pixels of upper edge to current area, otherwise we would count it multiple times, so use faster cv.line function for lower edge as we don't need the pixels idx array returned
                pixel_count_upper = len(a1)-2 #-2 as we paint on top of edge pixels
                cut_jet_img[a1, b1] = 255
                #calculate the mid point of the area we try to fill 
                iy2 = int((midline_coordinates[0][last_id]+midline_coordinates[0][id])/2) - min_y
                ixa2 = np.where(cut_jet_img[iy2,:] == 255)
                ix2 = int((ixa2[0][0]+ixa2[0][-1])/2)
                #flood fill the area 
                fill_area = cv.floodFill(cut_jet_img, None , (ix2, iy2), 255)[0]  #might be too slow, lets see, could be done with findcontours too
                #tv(cut_jet_img_copy)
                #count the nonzero pixels => area and scale with pixel size, and add previous values 
                area = ((pix_count_left_edge+pix_count_right_edge+pixel_count_upper+fill_area) * surface_scaler)
                area_sum += area
            # print("non zero area, pix count area:", np.count_nonzero(cut_jet_img_copy), (pix_count_left_edge+pix_count_right_edge+pixel_count_upper+pixel_count_lower+area))
                jet_areas.append(area) # convert to um2 and append to list
                p1x_last = p1x
                p1y_last = p1y
                p2x_last = p2x
                p2y_last = p2y
                last_id = id
                last_left_edge_idx = left_edge_idx
                last_right_edge_idx = right_edge_idx
        
        if last_id != len(midline_coordinates[0])-1: 
            id = -1 #=> get last entry, calc area until end of jet
            cut_jet_img = np.zeros((max_y-min_y, max_x-min_x), dtype="uint8") 
            #draw part of the edge contour for next section
            cut_jet_img[(left_edge_coordinates[0][last_left_edge_idx:]-min_y ,left_edge_coordinates[1][last_left_edge_idx:]-min_x)] = 255
            cut_jet_img[(right_edge_coordinates[0][last_right_edge_idx:]-min_y ,right_edge_coordinates[1][last_right_edge_idx:]-min_x)] = 255
            pix_count_left_edge = len(left_edge_coordinates[0][last_left_edge_idx:])
            pix_count_right_edge = len(right_edge_coordinates[0][last_right_edge_idx:])
            #calculate points of lower edge connection point, keep offset for cut image in mind
            p1x = left_edge_coordinates[1][-1] - min_x
            p1y = left_edge_coordinates[0][-1] - min_y
            p2x = right_edge_coordinates[1][-1] - min_x
            p2y = right_edge_coordinates[0][-1] - min_y
            #draw the upper and lower edge connection line
            a1, b1 = skimage.draw.line(p1y_last, p1x_last, p2y_last, p2x_last)
            cv.line(cut_jet_img, (p1x,p1y),(p2x,p2y), 255, thickness=1 ,lineType=cv.LINE_4) #we only count pixels of upper edge to current area, otherwise we would count it multiple times, so use faster cv.line function for lower edge as we don't need the pixels idx array returned
            pixel_count_upper = len(a1)-2 #-2 as we paint on top of edge pixels
            cut_jet_img[a1, b1] = 255
            #calculate the mid point of the area we try to fill 
            iy2 = int((midline_coordinates[0][last_id]+midline_coordinates[0][id])/2) - min_y
            ixa2 = np.where(cut_jet_img[iy2,:] == 255)
            ix2 = int((ixa2[0][0]+ixa2[0][-1])/2)
            #flood fill the area 
            fill_area = cv.floodFill(cut_jet_img, None , (ix2, iy2), 255)[0] #undocumented return value of cv.floodFill
            #count the nonzero pixels => area and scale with pixel size, and add previous values 
            area = ((pix_count_left_edge+pix_count_right_edge+pixel_count_upper+fill_area) * surface_scaler)
            area_sum += area
            #print("non zero area, pix count area:", np.count_nonzero(cut_jet_img_copy), (pix_count_left_edge+pix_count_right_edge+pixel_count_upper+pixel_count_lower+area))
        status = "Ok"
    except:
        status = "ERROR - Exception occured while calculating jet areas."
        jet_areas = [-1]
        area_sum = -1
        area_edge_indices = [[-1,-1]]


        
    return jet_areas, area_sum, area_edge_indices, status

def getSpinlineArea2_skimage_V4(midline_coordinates, left_edge_coordinates, right_edge_coordinates, area_positions, pixel_size=1, approx_dist=25):
    try:
            #find minimum and maximum x/y edge coordinates 
        min_x = min( np.min(left_edge_coordinates[1]), np.min(right_edge_coordinates[1])) 
        max_x = max( np.max(left_edge_coordinates[1]), np.max(right_edge_coordinates[1]))
        min_y = min( np.min(left_edge_coordinates[0]), np.min(right_edge_coordinates[0]))
        max_y = max( np.max(left_edge_coordinates[0]), np.max(right_edge_coordinates[0]))

        #to have a little distance from filled jet
        min_x = max(0,min_x-5)
        max_x = max_x+5 
        min_y = max(0,min_y-5)
        max_y = max_y+5 

        jet_areas = [[-1,-1] for i in range(len(area_positions))] #list of areas for given area positions
        area_edge_indices = [[-1,-1] for i in range(len(area_positions))] 
        last_id = 0
        area_sum = 0
        surface_scaler = np.square(pixel_size)
        last_left_edge_idx = 0
        last_right_edge_idx = 0
        p1x_last = left_edge_coordinates[1][0]-min_x
        p1y_last = left_edge_coordinates[0][0]-min_y
        p2x_last = right_edge_coordinates[1][0]-min_x
        p2y_last = right_edge_coordinates[0][0]-min_y

        area_positions_sorted = sorted(area_positions)
        area_positions_idx_t = [area_positions.index(i) for i in area_positions_sorted] #if we have two equal entries, we get the same index for all equal entries e.g. => [1,2,2,3] -> [0,1,1,2]
        area_positions_idx = area_positions_idx_t
        for i in range(1,len(area_positions_idx_t)):
            if area_positions_idx_t[i-1] == area_positions_idx_t[i]:
                area_positions_idx[i] += 1

        k = 0 
        last_i = -2
        for i in area_positions_sorted:
            if i == last_i: #if we have the same position two times, skip its area calc, and set area to 0, 
                last_i = i
                jet_areas[area_positions_idx[k]] = [k, 0]
                k += 1
                continue
            elif i == -1: #if invalid position
                jet_areas[area_positions_idx[k]] = [k, -1]
                k += 1
                continue
            else:
                last_i = i

            #take points before and after our mid-line point, and calculate slope
            if i < approx_dist: # if we are very close to our nozzle, we can't go further back
                id_b = 0
            else:
                id_b = i-approx_dist

            if (i+approx_dist) > len(midline_coordinates[0])-1 : # if we are very close to the end of our mid line, we can't go further 
                id_f = -1
            else:
                id_f = i+approx_dist+1 #+1 because of stupid slice indexing

            x = midline_coordinates[1][id_b:id_f] #+1 because of stupid slice indexing
            y = midline_coordinates[0][id_b:id_f]

            j = i-id_b # i in x, y is our reference midpoint through which the normal should pass 
            #print(id, id_b, id_f, i)       
            _, _, m, b = getlineregI(x, y, j) #get the slope and intersect for the approximated normal through the given points 

            #now we have to find the intersection of our normal with the contours, we can do so by checking if our points on the edge statisfy the y=mx+b equation of our normal line
            left_edge_idx = -1 #init to -1 so we can check wheter we found an intersection
            right_edge_idx = -1 

            for j in range(len(left_edge_coordinates[0])):
                #print("left edge test:", left_edge_coordinates[0][i] - m*left_edge_coordinates[1][i] + b) #debug only
                if ((m*left_edge_coordinates[1][j] + b) - left_edge_coordinates[0][j])  < 1: #as we have discrete x/y values for our edges, sometime we dont find perfect match, thus check if we are close -> lines might intersect between pixels 
                    #print("found intersection with left edge, idx:", i)
                    left_edge_idx = j
                    break #we can stop our for loop 

            for j in range(len(right_edge_coordinates[0])):
                if ((m*right_edge_coordinates[1][j] + b) - right_edge_coordinates[0][j]) < 1:
                    #print("found intersection with right edge, idx:", i)
                    right_edge_idx = j
                    break #we can stop our for loop 
            
            area_edge_indices[area_positions_idx[k]] = [left_edge_idx,right_edge_idx]
            
            #create closed jet image 
            cut_jet_img = np.zeros((max_y-min_y, max_x-min_x), dtype="uint8") 
            #draw part of the edge contour
            cut_jet_img[(left_edge_coordinates[0][last_left_edge_idx:left_edge_idx]-min_y ,left_edge_coordinates[1][last_left_edge_idx:left_edge_idx]-min_x)] = 255
            cut_jet_img[(right_edge_coordinates[0][last_right_edge_idx:right_edge_idx]-min_y,right_edge_coordinates[1][last_right_edge_idx:right_edge_idx]-min_x)] = 255
            pix_count_left_edge = len(left_edge_coordinates[0][last_left_edge_idx:left_edge_idx])
            pix_count_right_edge = len(right_edge_coordinates[0][last_right_edge_idx:right_edge_idx])
            #calculate points of lower edge connection point, keep offset for cut image in mind
            p1x= left_edge_coordinates[1][left_edge_idx] - min_x
            p1y = left_edge_coordinates[0][left_edge_idx] - min_y
            p2x = right_edge_coordinates[1][right_edge_idx] - min_x
            p2y = right_edge_coordinates[0][right_edge_idx] - min_y
            #draw the upper and lower edge connection line
            a1, b1 = skimage.draw.line(p1y_last, p1x_last, p2y_last, p2x_last)
            cv.line(cut_jet_img, (p1x,p1y),(p2x,p2y), 255, thickness=1 ,lineType=cv.LINE_4) #we only count pixels of upper edge to current area, otherwise we would count it multiple times, so use faster cv.line function for lower edge as we don't need the pixels idx array returned
            pixel_count_upper = len(a1)-2 #-2 as we paint on top of edge pixels
            cut_jet_img[a1, b1] = 255
            #calculate the mid point of the area we try to fill 
            iy2 = int((midline_coordinates[0][last_id]+midline_coordinates[0][i])/2) - min_y
            ixa2 = np.where(cut_jet_img[iy2,:] == 255)
            ix2 = int((ixa2[0][0]+ixa2[0][-1])/2)
            #flood fill the area 
            fill_area = cv.floodFill(cut_jet_img, None , (ix2, iy2), 255)[0]  #might be too slow, lets see, could be done with findcontours too
            #tv(cut_jet_img_copy)
            #count the nonzero pixels => area and scale with pixel size
            area = ((pix_count_left_edge+pix_count_right_edge+pixel_count_upper+fill_area) * surface_scaler)
            area_sum += area
            #
            jet_areas[area_positions_idx[k]] = [k, area]
            p1x_last = p1x
            p1y_last = p1y
            p2x_last = p2x
            p2y_last = p2y
            last_id = i
            last_left_edge_idx = left_edge_idx
            last_right_edge_idx = right_edge_idx
            k += 1

        if last_id != len(midline_coordinates[0])-1: 
            id = -1 #=> get last entry, calc area until end of jet
            cut_jet_img = np.zeros((max_y-min_y, max_x-min_x), dtype="uint8") 
            #draw part of the edge contour for next section
            cut_jet_img[(left_edge_coordinates[0][last_left_edge_idx:]-min_y ,left_edge_coordinates[1][last_left_edge_idx:]-min_x)] = 255
            cut_jet_img[(right_edge_coordinates[0][last_right_edge_idx:]-min_y ,right_edge_coordinates[1][last_right_edge_idx:]-min_x)] = 255
            pix_count_left_edge = len(left_edge_coordinates[0][last_left_edge_idx:])
            pix_count_right_edge = len(right_edge_coordinates[0][last_right_edge_idx:])
            #calculate points of lower edge connection point, keep offset for cut image in mind
            p1x = left_edge_coordinates[1][-1] - min_x
            p1y = left_edge_coordinates[0][-1] - min_y
            p2x = right_edge_coordinates[1][-1] - min_x
            p2y = right_edge_coordinates[0][-1] - min_y
            #draw the upper and lower edge connection line
            a1, b1 = skimage.draw.line(p1y_last, p1x_last, p2y_last, p2x_last)
            cv.line(cut_jet_img, (p1x,p1y),(p2x,p2y), 255, thickness=1 ,lineType=cv.LINE_4) #we only count pixels of upper edge to current area, otherwise we would count it multiple times, so use faster cv.line function for lower edge as we don't need the pixels idx array returned
            pixel_count_upper = len(a1)-2 #-2 as we paint on top of edge pixels
            cut_jet_img[a1, b1] = 255
            #calculate the mid point of the area we try to fill 
            iy2 = int((midline_coordinates[0][last_id]+midline_coordinates[0][id])/2) - min_y
            ixa2 = np.where(cut_jet_img[iy2,:] == 255)
            ix2 = int((ixa2[0][0]+ixa2[0][-1])/2)
            #flood fill the area 
            fill_area = cv.floodFill(cut_jet_img, None , (ix2, iy2), 255)[0] #undocumented return value of cv.floodFill
            #count the nonzero pixels => area and scale with pixel size, and add previous values 
            area = ((pix_count_left_edge+pix_count_right_edge+pixel_count_upper+fill_area) * surface_scaler)
            area_sum += area
            #print("non zero area, pix count area:", np.count_nonzero(cut_jet_img_copy), (pix_count_left_edge+pix_count_right_edge+pixel_count_upper+pixel_count_lower+area))
        status = "Ok"
    except:
        status = "ERROR - Exception occured while calculating jet areas."
        jet_areas = [[-1,-1]]
        area_sum = -1
        area_edge_indices = [[-1,-1]]

    return jet_areas, area_sum, area_edge_indices, status



#get jet diameter
def getFibreDiameterIntersectOLD(nozzle_mid_point, ref_distance, midline_coordinates, left_contour, right_contour, diameter_positions, pixel_size=1, relative=True, approx_dist=5):
    #tic = time.time()
    #print(len(midline_coordinates[0]))
    try:
        fibre_diameters = []
        midline_intersection_indices = []
        contour_intersection_indices = []
        for i in range(len(diameter_positions)):
            if relative == True:
                y_dist = np.rint((diameter_positions[i]/100)*ref_distance) # y_distance in pixels from nozzle for requested angle position in % of deposition distance
            else:
                y_dist = np.rint(diameter_positions[i]/pixel_size) #if we give absolute distance position in um

            #print("y_dist:", y_dist) #debug
            idx = np.where(midline_coordinates[0] == (y_dist + nozzle_mid_point[1])) #check index of the y-pos in our mid-line coordinates list
            if len(idx[0]) == 0: #if this position is not in our list
                fibre_diameters.append(-1) #append Nan if invalid y-pos, e.g. too low percentage, or too big/small absolute distance
                midline_intersection_indices.append(-1)
                contour_intersection_indices.append((-1, -1))
                continue #skip rest of current for-loop
            elif len(idx[0]) > 1: #more than one entry, can happen if mid-line runs horizontal for a short distance => staircase
                id = int(np.rint(np.mean(idx))) #take middle point of horizontal section, round to next integer
                midline_intersection_indices.append(id)
                #print("multi entries, id:", id)
            else: #if only one entry in list, take that one 
                id = idx[0][0]
                midline_intersection_indices.append(id)
            #print("id:", id)

            #now approximate mid_line with a line, over a short distance, so that we can easily get its slope and thus normal 
            
            
            #take points before and after our mid-line point, and calculate slope
            if id < approx_dist: # if we are very close to our nozzle, we can't go further back
                id_b = 0
            else:
                id_b = id-approx_dist

            if (id+approx_dist) > len(midline_coordinates[0])-1 : # if we are very close to the end of our mid line, we can't go further 
                id_f = -1
            else:
                id_f = id+approx_dist+1 #+1 because of stupid slice indexing

            x = midline_coordinates[1][id_b:id_f] #+1 because of stupid slice indexing
            y = midline_coordinates[0][id_b:id_f]
    
            #cv.line(img, (midx1,midy1), (midx2,midy2), color=255, thickness=2) #debug only, approx line
            #tv(img)
            ### CONTINUE HERE - FIX intersection
            #print(x, y)
            j = id-id_b # j in x, y is our reference midpoint through which the normal should pass 
            #print(id, id_b, id_f, i)       
            _, _, m, b = getlineregI(x, y, j) #get the slope and intersect for the approximated normal through the given points 

            #now we have to find the intersection of our normal with the contours, we can do so by checking if our points on the edge statisfy the y=mx+b equation of our normal line
            left_edge_idx = -1 #init to -1 so we can check wheter we found an intersection
            right_edge_idx = -1 


            for j in range(len(left_contour[0])):
                #print("left edge test:", left_edge_coordinates[0][i] - m*left_edge_coordinates[1][i] + b) #debug only
                if ((m*left_contour[1][j] + b) - left_contour[0][j])  < 1: #as we have discrete x/y values for our edges, sometime we dont find perfect match, thus check if we are close -> lines might intersect between pixels 
                    #print("found intersection with left edge, idx:", i)
                    left_edge_idx = j
                    break #we can stop our for loop 

            for j in range(len(right_contour[0])):
                if ((m*right_contour[1][j] + b) - right_contour[0][j]) < 1:
                    #print("found intersection with right edge, idx:", i)
                    right_edge_idx = j
                    break #we can stop our for loop 
                        
            #print(left_edge_idx) #debug only
            #print(right_edge_idx)
            #if we found intersection with both edges, calculate distance between points => is our diameter
            if (left_edge_idx != -1) and (right_edge_idx != -1):
                fibre_diameters.append(pixel_size * np.sqrt(np.square(left_contour[0][left_edge_idx]-right_contour[0][right_edge_idx])+np.square(left_contour[1][left_edge_idx]-right_contour[1][right_edge_idx])))
                contour_intersection_indices.append((left_edge_idx, right_edge_idx))
            else:
                fibre_diameters.append(-1)
                contour_intersection_indices.append((-1, -1))
        
        #deposition diameter directly from edges, here our line approach breaks due to vertical line
        last_diameter = pixel_size * np.sqrt(np.square(left_contour[0][-1]-right_contour[0][-1])+np.square(left_contour[1][-1]-right_contour[1][-1]))
        #toc = time.time()
        #print("diameter time", toc-tic)
        status = "Ok"
    except:
        status = "ERROR - Exception occured while caclulating fiber diameters."
        #print("ERROR - Exception occured while caclulating fiber diameters.")
        fibre_diameters = [-1]
        last_diameter = -1
        contour_intersection_indices = [(-1, -1)]
        midline_intersection_indices = [(-1, -1)]
    return fibre_diameters, last_diameter, contour_intersection_indices, midline_intersection_indices, status

def getFibreDiameterIntersect(nozzle_mid_point, midline_coordinates, left_contour, right_contour, mid_line_indices, pixel_size=1, approx_dist=5):
    #tic = time.time()
    #print(len(midline_coordinates[0]))
    p = 0
    try:
        fibre_diameters = []
        contour_intersection_indices = []
        for i in mid_line_indices:
            if i == -1: # if invalid midline_index
                fibre_diameters.append(-1) #append -1 if invalid y-pos, e.g. too low percentage, or too big/small absolute distance
                contour_intersection_indices.append((-1, -1))
            else:
                #take points before and after our mid-line point, and calculate slope
                if i < approx_dist: # if we are very close to our nozzle, we can't go further back
                    id_b = 0
                else:
                    id_b = i-approx_dist

                if (i+approx_dist) > len(midline_coordinates[0])-1 : # if we are very close to the end of our mid line, we can't go further 
                    id_f = None #stupid slice indexing -> gives us id_b: -> includes last element
                else:
                    id_f = i+approx_dist+1 #+1 because of stupid slice indexing

                x = midline_coordinates[1][id_b:id_f] 
                y = midline_coordinates[0][id_b:id_f]
        
                #cv.line(img, (midx1,midy1), (midx2,midy2), color=255, thickness=2) #debug only, approx line
                #tv(img)
                j = i-id_b # j in x, y is our reference midpoint through which the normal should pass     
                _, _, m, b = getlineregI(x, y, j) #get the slope and intersect for the approximated normal through the given points 
                
                # if p == 0:
                #     print(x)
                #     print(y)
                #     print(j, x[j], y[j])
                #     xm = x[j]
                #     ym = y[j]
                #     print(m, b)
                    
                #now we have to find the intersection of our normal with the contours, we can do so by checking if our points on the edge statisfy the y=mx+b equation of our normal line
                left_edge_idx = -1 #init to -1 so we can check wheter we found an intersection
                right_edge_idx = -1 

                for j in range(len(left_contour[0])):
                    #print("left edge test:", left_edge_coordinates[0][i] - m*left_edge_coordinates[1][i] + b) #debug only
                    if ((m*left_contour[1][j] + b) - left_contour[0][j])  < 1: #as we have discrete x/y values for our edges, sometime we dont find perfect match, thus check if we are close -> lines might intersect between pixels 
                        #print("found intersection with left edge, idx:", i)
                        left_edge_idx = j
                        break #we can stop our for loop 

                for j in range(len(right_contour[0])):
                    if ((m*right_contour[1][j] + b) - right_contour[0][j]) < 1:
                        #print("found intersection with right edge, idx:", i)
                        right_edge_idx = j
                        break #we can stop our for loop 
                            
                #print(left_edge_idx) #debug only
                #print(right_edge_idx)
                #if we found intersection with both edges, calculate distance between points => is our diameter
                if (left_edge_idx != -1) and (right_edge_idx != -1):
                    fibre_diameters.append(pixel_size * np.sqrt(np.square(left_contour[0][left_edge_idx]-right_contour[0][right_edge_idx])+np.square(left_contour[1][left_edge_idx]-right_contour[1][right_edge_idx])))
                    contour_intersection_indices.append((left_edge_idx, right_edge_idx))
                else:
                    fibre_diameters.append(-1)
                    contour_intersection_indices.append((-1, -1))

                # if p == 0:
                #     img = np.zeros((2048, 2448, 1), dtype=np.uint8)
                #     cv.drawMarker(img, (left_contour[1][left_edge_idx], left_contour[0][left_edge_idx]), color=255, thickness=2, markerSize=5, markerType=cv.MARKER_STAR)
                #     cv.drawMarker(img, (right_contour[1][right_edge_idx], right_contour[0][right_edge_idx]), color=255, thickness=2, markerSize=5, markerType=cv.MARKER_STAR)
                #     cv.drawMarker(img, (xm, ym), color=255, thickness=3, markerSize=5, markerType=cv.MARKER_CROSS)
                #     cv.line(img, (left_contour[1][left_edge_idx], left_contour[0][left_edge_idx]), (right_contour[1][right_edge_idx], right_contour[0][right_edge_idx]), color=150, thickness=2, lineType=cv.LINE_AA )
                #     tv(img, size=0.5)
                #     p += 1


        #deposition diameter directly from edges, here our line approach breaks due to vertical line
        last_diameter = pixel_size * np.sqrt(np.square(left_contour[0][-1]-right_contour[0][-1])+np.square(left_contour[1][-1]-right_contour[1][-1]))
        #toc = time.time()
        #print("diameter time", toc-tic)
        status = "Ok"
    except:
        status = "ERROR - Exception occured while caclulating fiber diameters."
        #print("ERROR - Exception occured while caclulating fiber diameters.")
        fibre_diameters = [-1]
        last_diameter = -1
        contour_intersection_indices = [(-1, -1)]

    return fibre_diameters, last_diameter, contour_intersection_indices, status



#get diameters at approx_dist spacing along midline, starting nozzle_dist pixels below nozzle
def getFibreDiameterAlongMidline(nozzle_mid_point, midline_coordinates, left_contour, right_contour, measure_dist=1, approx_dist=25, nozzle_dist = 5):
    #tic = time.time()
    
    try:
        fibre_diameters = []
        midline_intersection_indices = []
        contour_intersection_indices = []
        midline_coordinates_length = len(midline_coordinates[0])
        #print(midline_coordinates_length)
        #as contour and midline start/end segments are sometimes a little off, 
        # start a few pixels below the nozzle (nozzle_dist)
        # first diameter is the distance between the left and right edge at that that z-position
        #print(nozzle_mid_point)
        y_start = nozzle_dist + nozzle_mid_point[1][0] #double check [0]
        
        idx_m = np.where(midline_coordinates[0] == y_start)[0][0]
        idx_l = np.where(left_contour[0] == y_start)[0][0]
        idx_r = np.where(right_contour[0] == y_start)[0][0]
        #print(idx_m, idx_l, idx_r)
        fibre_diameters.append(np.sqrt(np.square(left_contour[0][idx_l]-right_contour[0][idx_r])+np.square(left_contour[1][idx_l]-right_contour[1][idx_r])))
        midline_intersection_indices.append(idx_m)
        contour_intersection_indices.append((idx_l, idx_r))

        #cv.drawMarker(img, (midline_coordinates[1][idx_m], midline_coordinates[0][idx_m]), 100, markerSize=10, thickness= 3, markerType= cv.MARKER_CROSS)
        #tv(img,size=0.5)

        last_left_edge_idx = idx_l
        last_right_edge_idx = idx_r
        for i in range(idx_m+measure_dist,midline_coordinates_length-1,measure_dist):  
           # print("idx_m:", i)
                      
            #now approximate mid_line with a line, over a short distance, so that we can easily get its slope and thus normal 
            #take points before and after our mid-line point, and calculate slope
            if i < approx_dist: # if we are very close to our nozzle, we can't go further back
                id_b = 0
            else:
                id_b = i-approx_dist

            if (i+approx_dist) > midline_coordinates_length-1 : # if we are very close to the end of our mid line, we can't go further 
                id_f = -1
            else:
                id_f = i+approx_dist+1 #+1 because of stupid slice indexing

            x = midline_coordinates[1][id_b:id_f] #+1 because of stupid slice indexing
            y = midline_coordinates[0][id_b:id_f]
    
            #cv.line(img, (midx1,midy1), (midx2,midy2), color=255, thickness=2) #debug only, approx line
            #tv(img)
            ### CONTINUE HERE - FIX intersection
            #print(x, y)
            j = i-id_b # j in x, y is our reference midpoint through which the normal should pass 
            #print(id, id_b, id_f, i)       
            _, _, m, b = getlineregI(x, y, j) #get the slope and intersect for the approximated normal through the given points 

            #now we have to find the intersection of our normal with the contours, we can do so by checking if our points on the edge statisfy the y=mx+b equation of our normal line
            left_edge_idx = -1 #init to -1 so we can check wheter we found an intersection
            right_edge_idx = -1 


            for j in range(len(left_contour[0])):
                #print("left edge test:", left_edge_coordinates[0][i] - m*left_edge_coordinates[1][i] + b) #debug only
                if ((m*left_contour[1][j] + b) - left_contour[0][j])  < 1: #as we have discrete x/y values for our edges, sometime we dont find perfect match, thus check if we are close -> lines might intersect between pixels 
                    #print("found intersection with left edge, idx:", i)
                    left_edge_idx = j
                    break #we can stop our for loop 

            for j in range(len(right_contour[0])):
                if ((m*right_contour[1][j] + b) - right_contour[0][j]) < 1:
                    #print("found intersection with right edge, idx:", i)
                    right_edge_idx = j
                    break #we can stop our for loop 
                        
            #print(left_edge_idx) #debug only
            #print(right_edge_idx)
            #if we found intersection with both edges, calculate distance between points => is our diameter
   

            if (left_edge_idx != -1) and (right_edge_idx != -1) and (last_left_edge_idx < left_edge_idx) and (last_right_edge_idx < right_edge_idx):
                fibre_diameters.append(np.sqrt(np.square(left_contour[0][left_edge_idx]-right_contour[0][right_edge_idx])+np.square(left_contour[1][left_edge_idx]-right_contour[1][right_edge_idx])))
                contour_intersection_indices.append((left_edge_idx, right_edge_idx))
                midline_intersection_indices.append(i)
                last_right_edge_idx = right_edge_idx
                last_left_edge_idx = left_edge_idx
                #cv.drawMarker(img, (midline_coordinates[1][i], midline_coordinates[0][i]), 100, markerSize=10, thickness= 3, markerType= cv.MARKER_CROSS)
                #cv.drawMarker(img, (left_contour[1][left_edge_idx], left_contour[0][left_edge_idx]), 100, markerSize=10, thickness= 3, markerType= cv.MARKER_CROSS)
                #cv.drawMarker(img, (right_contour[1][right_edge_idx], right_contour[0][right_edge_idx]), 100, markerSize=10, thickness= 3, markerType= cv.MARKER_CROSS)
                #tv(img,size=0.5)
    
            #print("dia:", fibre_diameters[-1])
            #print("idx", contour_intersection_indices[-1])

        #print(len(contour_intersection_indices))
        
        #tv(img,size=0.5)

        #toc = time.time()
        #print("diameter time", toc-tic)
        status = "Ok"
    except:
        status = "ERROR - Exception occured while caclulating fiber diameters."
        #print("ERROR - Exception occured while caclulating fiber diameters.")
        fibre_diameters = [-1]
        contour_intersection_indices = [(-1, -1)]
        midline_intersection_indices = [-1]

    return fibre_diameters, contour_intersection_indices, midline_intersection_indices, status

#get volumes along midline with segmentation following fibre_diameters list 
def getSpinlineVolumes(left_edge_coordinates, right_edge_coordinates, fibre_diameters, contour_intersection_indices, pixel_size=1):
    
    #sanity check
    #print("len dia list", len(fibre_diameters))
    #print("len edge idx list", len(contour_intersection_indices))


    #move outside if performance hit too large 
    def calcCylinderVolume(a, b, d, dia1): #tested ok
        # a,b,d points in our counter-clockwise count direction
        #h =   np.abs((d[0]-a[0])*(a[1]-b[1])-(a[0]-b[0])*(d[1]-a[1])) / np.sqrt(np.square(d[0]-a[0])+np.square(d[1]-a[1])) #calculate height of cylinder, is equal to distance of point B (or C) to line defined by point A and D
        h = np.sqrt(np.square(b[0]-a[0])+np.square(b[1]-a[1])) #this should actually be sufficient, as we have a cylinder line a-b, and b-c are parallel, so height is equal to distance between a an b
        v = h * (np.pi*np.square(dia1/2)) #calc volume = height*area , area is assumed to be circle 
        return v
    
    def caclConeVolumeFourPoints(a, b, c, d, dia1, dia2): #tested, ok
        # First, find intersection point of the two lines that are running between AB and DC 
        # use vector form of lines, then form linear system and solve with cramers rule 
        u1 = b[0]-a[0] #x - of vector from point A to B # B hernehmen A abziehen
        v1 = b[1]-a[1] #y - of vector from point A to B 
        u2 = c[0]-d[0] #x - of vector from point D to C
        v2 = c[1]-d[1] #y - of vector from point D to C
        xd = d[0]-a[0] # x difference point D and A, follows from linear system 
        yd = d[1]-a[1] # y difference point D and A, follows from linear system 
        matd = np.array([[u1,-u2],[v1,-v2]]) #matrix D, => linear system matrix 
        matd1 = np.array([[xd,-u2],[yd,-v2]]) #matrix D1 with swaped first column, following cramers rule
        det1 = np.linalg.det(matd) 
        if det1 == 0: #sometimes det1 can be 0 => almost paralell lines, even though dia1 != dia2 (staricase/rounding errors), in that case following calculations will fail (division 0), fall back to cylinder approximation
            v = calcCylinderVolume(a, b, d, dia1)
            #print("fallback cylinder") #debug only
            return v  
        p = np.linalg.det(matd1)/det1 # find a by calculating/dividing the determinats, cramers rule


        xI = a[0] + p * u1 #find x of intersection point by plugging in a in line equation
        yI = a[1] + p * v1 #find y of intersection point by plugging in a in line equation
        
        #print("points:", a, b, c, d) #debug only
        #print("diameters:", dia1, dia2, dia1-dia2) #debug only
        #print("mat1:", matd)
        #print("mat2:", matd1)
        #print("xI:", xI) #debug only
        #print("yI:", yI) #debug only

        # now calculate the height of lines running between AD and BC, respectively, to the intersection point 
        h1 = np.abs((d[0]-a[0])*(a[1]-yI)-(a[0]-xI)*(d[1]-a[1])) / np.sqrt(np.square(d[0]-a[0])+np.square(d[1]-a[1]))
        h2 = np.abs((c[0]-b[0])*(b[1]-yI)-(b[0]-xI)*(c[1]-b[1])) / np.sqrt(np.square(c[0]-b[0])+np.square(c[1]-b[1]))

        #print("h1:", h1) #debug only
        #print("h2:", h2) #debug only


        # now calculate volume of the two cones 
        v1 = h1 * (np.pi*np.square(dia1/2)) # cone between AD and intersection point, assuming circular base
        v2 = h2 * (np.pi*np.square(dia2/2))#cone between BC and intersecttion point, assuming circular base
        v = abs(v1-v2) # jet section is difference of the two volumes
        #depending on the diameters and point positions a number of cases in terms of cone 
        # orientation can occure, however we always have to get the difference between the two volumes
        # as we always try to cut away the smaller volume from the bigger volume 


        #print("ERROR - Negative volume, cone calculation") #debug only
        # print("points:", a, b, c, d) #debug only
        # print("diameters:", dia1, dia2, dia1-dia2) #debug only
        # print("h1:", h1) #debug only
        # print("h2:", h2) #debug only
        # #print("mat1:", matd)
        # #print("mat2:", matd1)
        # #print("xI:", xI) #debug only
        # #print("yI:", yI) #debug only
        # print("v1:", v1)
        # print("v2:", v2)
        # print("v:", v)
        if v < 0:
            print("ERROR - Negative volume, cone calculation") #debug only

        return v
    
    def calcFourthPointCone(a,d,e, side): #tested, ok
        #a, d are points of cone base
        # e is point of one cone leg that need to be mirrored
        # side specifies if leg is between ae (side = l) or de (side = r) 

        f = np.array([0,0])
        #First calculate normal vector for line between ab 
        n = np.array([-(a[1]-d[1]),a[0]-d[0]])  # with dx = a[0]-b[0], dy=a[1]-b[1], rotmatrix for 2D 90deg => n = [-dy, dx]

        #Now calculate vector ag or dg 
        if side == "l":
            v1 = e-a # E hernehmen A abziehen
        elif side == "r":
            v1 = e-d
        else:
            print("Error, calculating fourth point for cone, invalid side argument")
            return f

        # Now calculate orthogonal projection of v1 onto n 
        v2 = (np.dot(v1,n) / np.square(np.linalg.norm(n))) * n #can be looked up in google 

        # Now calculate the mirrored v1 vector 
        v3 = 2 * v2 - v1   # see paper documentation or google search vector reflection, e.g. https://math.stackexchange.com/questions/3301455/reflect-a-2d-vector-over-another-vector-without-using-normal-why-this-gives-the

        #Finally calculate point d 
        if side == "l":
            f = d + v3 #go from point d along vector v3 
        else:
            f = a + v3  #go from point a along cector v3 

        return f
    
    #init once, should make for loop bit faster
    a = np.array([0,0], dtype="float")
    b = np.array([0,0], dtype="float")
    c = np.array([0,0], dtype="float")
    d = np.array([0,0], dtype="float")
    volume_conversion = pixel_size**3 # pixel size^3 to get volume conversion factor 
    all_volumes = []
    total_volume = 0
    try:
        for i in range(len(contour_intersection_indices)-1): # len()-1 to prevent i+1 to go out of bounds => all_volumes is one shorter than volume_points list
            #print(contour_intersection_indices[i])
            nl = int(contour_intersection_indices[i][0]) #nl is index of point on left edge, with index i 
            nr = int(contour_intersection_indices[i][1]) #nr is index of point on right edge, with index i 
            ml = int(contour_intersection_indices[i+1][0]) #ml is index of point on left edge, with index i+1
            mr = int(contour_intersection_indices[i+1][1]) #ml is index of point on right edge, with index i+1

            #print("nl, nr:", nl, nr) #debug only
            #print("ml, mr:", ml, mr) #debug only

            a[0] = left_edge_coordinates[1][nl] # x coordinate point a                    #         a------------d
            a[1] = left_edge_coordinates[0][nl] # y coordinate point a                    #          \          /
            b[0] = left_edge_coordinates[1][ml] # x coordinate point b                    #           \        /  
            b[1] = left_edge_coordinates[0][ml] # y coordinate point b                    #            b------c
            c[0] = right_edge_coordinates[1][mr] # x coordinate point c                    # it can happen that dia2 is bigger dia1
            c[1] = right_edge_coordinates[0][mr] # y coordinate point c                    # in that case the cone is flipped:  (this is handled in the cone volume function)
            d[0] = right_edge_coordinates[1][nr] # x coordinate point d                    #    a------d   
            d[1] = right_edge_coordinates[0][nr] # y coordinate point d                    #   /        \             
             
                                                                                           #  /          \
            #print("points, selected:", a, b, c, d) #debug only                            # b------------c

            dia1 = fibre_diameters[i] #distance between ad , IN PIXELS, NOT um !
            dia2 = fibre_diameters[i+1] #distance between bc


            if dia1 == dia2: # if the two diameters we look at are the same, we approximate as a cylinder 
                v = calcCylinderVolume(a, b, d, dia1)
                #print(a,b,dia1)
                #print("initial cylinder") #debug only
            else: #if they are not the same, approximate as a cone
            # print("cone calc")
                if (a==b).all(): #if we have "triangle" case => cone cut off at its base, so a and b or c and d fall together 
                    b = calcFourthPointCone(a,d,c, "r") #calc our fourth point by mirroring one leg to the other side 
                elif  (c==d).all(): #other way around
                    c = calcFourthPointCone(a,d,b, "l")
                v = caclConeVolumeFourPoints(a, b, c, d, dia1, dia2)
            
            v = v * volume_conversion #conversion to um3
            total_volume += v #add up total volume
            all_volumes.append(v) 
            #all_volumes.append((v,a.copy(),b.copy(),c.copy(),d.copy()))  #we have to use .copy() because otherwise we give it a "poitner"(or pythons equivalent), so we end with the same (last) point in every list enty
        #print(len(all_volumes))
        #Returns:
        #all_volumes => list of shape (volume in um3, x,y_coord point a, x,y_coord point b, x,y_coord point c, x,y_coord point d, x,y_coord midpoint ad, x,y_coord midpoint bc)
        # total_volume => volume of complete jet in um3
        status = "Ok"
    except:
        status = "ERROR - Exception occured while caclulating jet volumes."
        all_volumes = [-1]
        total_volume = -1

    return all_volumes, total_volume, status


### Functions for Initialisation and Nozzle position updates
def getNozzleContour(cimg, window_resize_scaler=1, Clow_Threshold=50, Cheight_Threshold=150, Ckernel_size=3, HLrho=1, HLtheta=(np.pi/180), HLthreshold=50, HLminLineLength=50, HLmaxLineGap=10, VerticalAngleThres=88):
    #tv(image) #debug only
    #image = cv.GaussianBlur(image,(7,7),10) #some blur can sometimes help to better detect the edges
    #tv(image) #debug only
    canny_edges_img = cv.Canny(cimg, Clow_Threshold, Cheight_Threshold, None, Ckernel_size) #detect edges using canny
    #tv(canny_edges_img) #debug only
    #timg = cv.cvtColor(canny_edges_img, cv.COLOR_GRAY2BGR) #copy for debug in BGR space

    #lets use probabilistic hough wich gives us the end points, so we dont have to calculate them 
    # It gives as output the extremes of the detected lines (x0,y0,x1,y1), however does not seem to be sorted
    lines = cv.HoughLinesP(canny_edges_img, rho=HLrho, theta=HLtheta, threshold=HLthreshold, minLineLength=HLminLineLength, maxLineGap=HLmaxLineGap) 
    
    # If lines have been found, go through list and do some operations to find the two lines that best describe our nozzle edge 
    # Should be re-structured to avoid nested if's
    if lines is not None: 
        if len(lines) >= 2: #make sure there are at least two lines detected
            lines = sortLinesLeftToRightTopPointFirst(lines) #sort lines from left to right, and have top point in first position of vector    
            line_idx = [] #create a list for line array index for lines that are vertical, only vertical lines can represent our left and right nozzle edge          

            for i in range(len(lines)): #go through the list of lines
                l = lines[i][0] #pick the vector that represents (x0,y0,x1,y1), for some reason lines structure is [[[x0.0 y0.0 x1.0 y1.0]][[x0.1 y0.1 x1.1 y1.1]]]
                #cv.line(timg, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv.LINE_AA) #draw all lines, for debugging
                #cv.drawMarker(timg, (l[0], l[1]), (255,0,0), markerType = cv.MARKER_SQUARE, markerSize= 20) #should be top, debug only
                #cv.drawMarker(timg, (l[2], l[3]), (0,255,0), markerType = cv.MARKER_TILTED_CROSS, markerSize= 20) #should be bottom, debug only
                #tv(timg) #debuggin only
                if abs(np.arctan2((l[3]-l[1]),(l[2]-l[0])) *180 /np.pi  ) > VerticalAngleThres: #if the angle of the line is greater vertical_angle_thres, -> look for (almost) vertical lines
                    #cv.line(timg, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv.LINE_AA) #draw the vertical line, debugging only
                    line_idx.append(i) #save wich lines are the vertical ones
                    #tv(timg) #debuggin only

        
        else: # if there are not at least two lines detected 
            print("WARNING: LESS THAN TWO LINES DETECTED.")
            return 0, 0, 0

        #print(len(line_idx)) #debug only
        #print(line_idx) #debug only
        if len(line_idx) >= 2: #Make sure there are at least two verical lines detected 
        
            #select the correct vertical lines => find lines with lowest lower point => closest to the corner with lower, horizontal nozzle edge 
            mid_X = 0 # mid point of lower X-Coordinates of all vertical lines
            for i in range(len(line_idx)):
                l = lines[i][0] #pick the vector that represents (x0,y0,x1,y1), for some reason lines structure is [[[x0.0 y0.0 x1.0 y1.0]][[x0.1 y0.1 x1.1 y1.1]]]
                mid_X = mid_X + l[2] #Add up all X-Coordinate values 
            mid_X = int(mid_X/len(line_idx)) #Divide by number of coordinates 

            PLL = 0 #Y-Point coordinate lowest left
            PLR = 0 #Y-Point coordinate lowest rigth 
            for i in range(len(line_idx)): 
                l = lines[i][0] #pick the vector that represents (x0,y0,x1,y1), for some reason lines structure is [[[x0.0 y0.0 x1.0 y1.0]][[x0.1 y0.1 x1.1 y1.1]]]
                if l[2] < mid_X and l[3] >= PLL:  #if X-Coordinate of lower point of current line is smaller mid_X, it is left of middle point, if it got a bigger Y-Value than previously checked points=> remember its index
                    PLL = l[3] #store new lowest point => highest Y Value 
                    PLL_idx = line_idx[i] #store corresponding index => so we know which line it belongs to 
                if l[2] > mid_X and l[3] >= PLR: #if X-Coordinate of lower point of current line is bigger mid_X, it is rigth of middle point, if it got a bigger Y-Value than previously checked points=> remember its index
                    PLR = l[3] #store new lowest point => highest Y Value
                    PLR_idx = line_idx[i] #store corresponding index => so we know which line it belongs to 
            
            #print("PLL:", PLL, "PLL_idx:", PLL_idx, "PLR:", PLR, "PLR_idx:", PLR_idx) #debugging only

            #pick the two lines that best represent our left/right nozzle edge
            lL = lines[PLL_idx][0] #pick the vector that represents (x0,y0,x1,y1) for left line
            lR = lines[PLR_idx][0] # and for right line

            #img1 = cv.cvtColor(image.copy(),cv.COLOR_GRAY2BGR) #copy and convert to BGR color space so that we can have colored markers/lines
            height = cimg.shape[0] #get image height so that we can set trackbar limits
            width= cimg.shape[1]
            cv.namedWindow('Nozzle position initialisation', cv.WINDOW_GUI_EXPANDED) # make a window with name 'image', use expanded gui to allow re-sizing via dragging
            cv.resizeWindow("Nozzle position initialisation", width=int(width*window_resize_scaler), height=int(height*window_resize_scaler)) #not pretty but ok for now, give it a size that works 
            cv.createTrackbar('Left', 'Nozzle position initialisation', lL[3], height, callback) #trackbar, init to left point lower Y coordinate
            cv.createTrackbar('Right', 'Nozzle position initialisation', lR[3], height, callback) #trackbar, init to right point lower Y coordinate

            #calc m and b for y=mx+b for our two vertical lines
            #Left line - Extended over whole image
            if lL[0] == lL[2]: #if both X-Values are the same -> perfect vertical line, y=mx+b not valid, easy to find points at image edge
                #PL_h => highest Point, at top of image; PL_l => lowest Point, at bottom of image
                PL_hx = lL[0] #X coordinate is always same, as given by line 
                PL_hy = 0 #y coordinate is zero, upper image edge
                PL_lx = lL[0] #X coordinate is always same, as given by line 
                PL_ly = height #y coordinate is heigth of image
                Lv = True #flag that line is perfect vertical line
            else:
                m1 = int((lL[3]-lL[1])/(lL[2]-lL[0])) # m = (y1-y0)/(x1-x0)
                b1 = int(lL[2]-m1*lL[0]) #b = yi-m*xi 
                PL_hx = int(-b1/m1) #calculate X coordinate for highest point, at which y=0 
                PL_hy = 0 #y coordinate is zero, upper image edge
                PL_lx = int((height-b1)/m1) #calculate X coordinate for lowest point, at which y = height of image
                PL_ly = height # y coordinate is heigth of image
                Lv = False #flag that line is not perfect vertical line
            
            #Right line - Extended over whole image
            if lR[0] == lR[2]: #if both X-Values are the same -> perfect vertical line, y=mx+b not valid, easy to find points at image edge
                #PR_h => highest Point, at top of image; PR_l => lowest Point, at bottom of image
                PR_hx = lR[0] #X coordinate is always same, as given by line 
                PR_hy = 0 #y coordinate is zero, upper image edge
                PR_lx = lR[0] #X coordinate is always same, as given by line 
                PR_ly = height #y coordinate is heigth of image
                Rv = True #flag that line is not perfect vertical line
            else:
                m2 = int((lR[3]-lR[1])/(lR[2]-lR[0])) # m = (y1-y0)/(x1-x0)
                b2 = int(lR[2]-m2*lR[0]) #b = yi-m*xi 
                PR_hx = int(-b2/m2) #calculate X coordinate for highest point, at which y=0 
                PR_hy = 0 #y is zero, upper image edge
                PR_lx = int((height-b2)/m2) #calculate X coordinate for lowest point, at which y = height of image
                PR_ly = height # y is heigth of image
                Rv = False #flag that line is not perfect vertical line
                
            PL_oy = 0 #variables for old pixel coordinates, so that we know when we have to re-draw our image 
            PR_oy = 0 #init with 0 so that in first while loop we draw everything 


            while(1): #keep in this loop until we are happy with our selection 
                PL_ny = cv.getTrackbarPos('Left', 'Nozzle position initialisation') #get out current trackbar values, is new y-coordinate of point
                PR_ny = cv.getTrackbarPos('Right', 'Nozzle position initialisation')

                if PL_ny != PL_oy or PR_ny != PR_oy : #if any of the values have changed, re-draw our lines, points etc
                    cimg1 = cimg.copy()
                    #img1 = cv.cvtColor(image.copy(),cv.COLOR_GRAY2BGR) #first make new copy, since we can't delete old lines/points
                    #No text for now, needs proper scaling, maybe implement later
                    #img1 = cv.putText(img1, "Adjust Sliders to tune lower nozzle edge position.", org=(10,10), fontFace=cv.FONT_HERSHEY_PLAIN, fontScale=font_scale, color=(255,255,0), thickness=font_thickness, lineType=cv.LINE_AA) #put some text into the image to give instructions 
                    #img1 = cv.putText(img1, "Press 'enter' to finish selection.", org=(10,10), fontFace=cv.FONT_HERSHEY_PLAIN, fontScale=font_scale, color=(255,255,0), thickness=font_thickness, lineType=cv.LINE_AA) #put some text into the image to give instructions 
                    PL_oy = PL_ny #store new values for next loop and calculations
                    PR_oy = PR_ny 
                    
                    #Draw lines of nozzle edges
                    cv.line(cimg1, (PL_hx,PL_hy), (PL_lx,PL_ly),(0,100,255),thickness=1)#left edge 
                    cv.line(cimg1, (PR_hx,PR_hy), (PR_lx,PR_ly),(0,100,255),thickness=1)#right edge
                    
                    #Calculate Point coordinate on left line and draw Marker
                    if Lv ==True: #if we have perfect vertical line
                        PL_x = PL_hx # always same X
                    else:
                        PL_x = int((PL_oy-b1)/m1) #calc X-Coordinate for current X coordinate 
                    cv.drawMarker(cimg1, (PL_x,PL_oy), (255,0,0), markerType=cv.MARKER_TILTED_CROSS, markerSize=5, thickness=1)
                    
                    #Calculate Point coordinate on right line and draw Marker
                    if Rv ==True: #if we have perfect vertical line
                        PR_x = PR_hx # always same X
                    else:
                        PR_x = int((PR_oy-b2)/m2) #calc X-Coordinate for current X coordinate 
                    cv.drawMarker(cimg1, (PR_x,PR_oy), (255,0,0), markerType=cv.MARKER_TILTED_CROSS, markerSize=5, thickness=1)
                    
                    #Draw line between left and right point => Nozzle lower edge
                    cv.line(cimg1, (PL_x,PL_oy), (PR_x,PR_oy),(0,0,255),thickness=1, lineType=cv.LINE_AA) 

                    #Calculate Nozzle-Mid point
                    Nozzle_midX = int(PL_x+(PR_x-PL_x)/2) #calculate coordinates of our line midpoint 
                    Nozzle_midY = int(PL_oy+(PR_oy-PL_oy)/2) #midpoint actually easier to calc with (x1+x2)/2 , dummy..
                    cv.drawMarker(cimg1, (Nozzle_midX, Nozzle_midY), (255,0,0), markerType = cv.MARKER_TILTED_CROSS, markerSize=5, thickness=1) #draw the mid point marker


                cv.imshow('Nozzle position initialisation', cimg1) #show the pictures 
                k = cv.waitKey(1) & 0xFF #0xFF has something to do with key codes 
                if k == 13: #if enter/return key is pressed, break out of while loop, means selection is done 
                    break  
            cv.destroyWindow('Nozzle position initialisation')
        
            #make it same structrue as other opencv related lists of points: [[x1,x2,x3],[y1,y2,y3]]
            left_line_points = [[PL_hx,PL_x],[PL_hy,PL_oy]] #from top of image down to intersection with lower nozzle edge line
            right_line_points = [[PR_hx,PR_x],[PR_hy,PR_oy]] #from top of image down to intersection with lower nozzle edge line
            nozzle_mid_point = [[Nozzle_midX],[Nozzle_midY]] #on lower nozzle edge line
        
        else: # if there are not at least two lines detected 
            print("WARNING: LESS THAN TWO VERTICAL LINES DETECTED.")
            return 0, 0, 0
    else:# if there are no lines detected 
        print("WARNING: NO LINES DETECTED.")
        return 0, 0, 0
    #tv(timg) #debug only   
    return left_line_points, right_line_points, nozzle_mid_point #return the relevant point coordinates found lines, the vertical line index list, and the mid points 

def SelectNozzleROI(img, window_size):
    r = (0,0,0,0) #tuple for ROI rectangle (), selectROI returns (Top_Left_X, TOP_Left_Y, Width, Height) of ROI
    while r == (0,0,0,0): #let us only proceed once we made a ROI selection, e.g. if we close the window without selecting a ROI it will be reopened again
        cv.namedWindow("Select ROI for Nozzle Detection", cv.WINDOW_GUI_EXPANDED) #creat window with GUI_EXPANDED, so that it can be resized by pulling on the windows edges 
        cv.resizeWindow("Select ROI for Nozzle Detection", width=window_size[0], height=window_size[1])  #resize a bit, since our camera makes very large images that don't fit on screen
        roi_image = cv.putText(img.copy(), "Select ROI. Press 'enter' to finish selection.", org=(50,50), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255,255,0), thickness=2, lineType=cv.LINE_AA) #put some text into the image to give instructions 
        r = cv.selectROI("Select ROI for Nozzle Detection", roi_image, printNotice=False) #open window wit ROI selection function, disable "printNotice" to console 
        cv.destroyWindow("Select ROI for Nozzle Detection") #when selection is done, close the window
    return r

def OffsetROI(coordinates, ROI, dimx=0, dimy=1): #Transforms the Point-Coordinates in "coordinates" by offsetting them with ROI 
    #ROI has structure: (Top_Left_X, TOP_Left_Y, Width, Height) 
    #Coordinates has structure(in case of list): [(x1,x2,x3),(y1,y2,y3)] , with xi,yi Points 
    #This is equivalent to a simple offset addition

    if isinstance(coordinates, list): #if we give it a list of structure: [(x1,x2,x3),(y1,y2,y3)]
        for i in range(len(coordinates[dimx])):
            coordinates[dimx][i] = coordinates[dimx][i] + ROI[0]
            coordinates[dimy][i] = coordinates[dimy][i] + ROI[1]

    if isinstance(coordinates, np.ndarray): #if we give it a numpy array of structure:
        for i in range(coordinates.shape[dimx]):
            coordinates[i][dimx] = coordinates[i][dimx] + ROI[0]
            coordinates[i][dimy] = coordinates[i][dimy] + ROI[1]

    return coordinates

def InitNozzlePosition(img, window_resize_scaler=1):
    window_size = (int(img.shape[1]/window_resize_scaler), int(img.shape[0]/window_resize_scaler)) #(width, heigth)        
    cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR) #make copy in BGR space so that we can have colored markers, lines, text etc.

    left_line_points = 0 #init variable for while loop, not elegant but works 
    while left_line_points == 0: 
        r = SelectNozzleROI(cimg, window_size) #first call function to select nozzle ROI, returns (Top_Left_X, TOP_Left_Y, Width, Height) of ROI  
        nozzle_ROI_image = cimg[r[1]:(r[1]+r[3]),r[0]:(r[0]+r[2])] #make croped image for nozzle contour detection; format:  source_image[ start_row : end_row, start_col : end_col]
        left_line_points, right_line_points, nozzle_mid_point = getNozzleContour(nozzle_ROI_image, window_resize_scaler) #call function to get nozzle contour, return values are relative to ROI coordinates 
    #Translate Coordinates from ROI into original image coordinates
    #This is equivalent to a simple offset addition
    left_line_points = OffsetROI(left_line_points, r)
    right_line_points = OffsetROI(right_line_points, r)
    nozzle_mid_point = OffsetROI(nozzle_mid_point, r)

    return left_line_points, right_line_points, nozzle_mid_point
