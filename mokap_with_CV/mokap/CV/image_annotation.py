import cv2 as cv 
import numpy as np 
from matplotlib import colormaps 

#colormaps / colors 
# rgb_spring = colormaps['spring'](np.linspace(0.0, 1.0, 100))[np.newaxis, :, :3] #gives 100 rgb value pairs of copper colormap
# bgr_spring = np.flip(rgb_spring[0], axis=1)*255

# mostly color blind save colors, from https://immunobiology.duke.edu/sites/default/files/2023-04/Colorblind-Palette.pdf

color_left_edge = (51,136,34) #bright green
color_right_edge = (51,153,153) # muted olive
color_midline = (119,51,238) ##vibrant magenta 

color_jet_angle = (187,119,0) #vibrant blue
color_diameter_lines = (51,119,238) #vibrant orange

color_jet_length = (221,221,221) #pale grey

color_area = (238,187,51) #vibrant cyan

color_volume = (68,204,187) #bright yellow

color_text = (170,119,68) # bright blue
color_text = (255,0,0) # bright blue

#### Visualisation functions

def mark(img, coordinates, color=(0,0,255), idx=-1, dimx=0, dimy=1, markersize=5, thickness = 2, markertype = cv.MARKER_CROSS):
    if isinstance(coordinates, list): #if we give it a list of structure: [(x1,x2,x3),(y1,y2,y3)]
        if idx == -1: #if idx = -1 : draw all markers
            for i in range(len(coordinates[dimx])): #go through list, draw a marker for each point
                cv.drawMarker(img, (coordinates[dimx][i], coordinates[dimy][i]), color, markerType = markertype, markerSize=markersize, thickness=thickness)
        else: #otherwise only the specified point 
            cv.drawMarker(img, (coordinates[dimx][idx], coordinates[dimy][idx]), color, markerType = markertype, markerSize=markersize, thickness=thickness)

        
    if isinstance(coordinates, np.ndarray): #if we give it a numpy array of structure:
        if idx == -1:
            for i in range(coordinates.shape[dimx]):
                cv.drawMarker(img, (coordinates[i][dimx], coordinates[i][dimy]), color, markerType = markertype, markerSize=markersize, thickness=thickness)
        else:
            cv.drawMarker(img, (coordinates[i][dimx], coordinates[i][dimy]), color, markerType = markertype, markerSize=markersize, thickness=thickness)
    
    
    return

#function to draw lines between pairs of points given in coordinates list/array
def lineBetweenPoints(img, coordinates, dimx=0, dimy=1, thickness=2, color=(0,0,255), linetype=cv.LINE_AA): #thickness 2
    if isinstance(coordinates, np.ndarray): #if we give it a numpy array of structure:
        coordinates = coordinates.tolist() #turn it into a list, so that we don't have to duplicate our code for arrays 

    if isinstance(coordinates, tuple): #if we give it a tuple structure: (array, array) or (list, list)
        a = coordinates[0]
        b = coordinates[1]
        if isinstance(a, np.ndarray): #if we give it a numpy array of structure:
            a = a.tolist() #turn it into a list, so that we don't have to duplicate our code for arrays
        if isinstance(b, np.ndarray): #if we give it a numpy array of structure:
            b = b.tolist() #turn it into a list, so that we don't have to duplicate our code for arrays

        coordinates = [a,b]  #turn it into one list
        #print("linebtp,coords:", coordinates)
    

    if isinstance(coordinates, list): #if we give it a list of structure: [(x1,x2,x3),(y1,y2,y3)]
        if len(coordinates[dimx]) == len(coordinates[dimy]) and len(coordinates[dimx]) >=2 : # make sure we have equal amount of x and y coordinates  
                for i in range(0,len(coordinates[dimx])-1,1): #
                    x1 = coordinates[dimx][i]
                    x2 = coordinates[dimx][i+1]
                    y1 = coordinates[dimy][i]
                    y2 = coordinates[dimy][i+1]
                    # print("read x1:", x1) #DEBUG ONLY
                    # print("read y1:", y1)
                    # print("read x2:", x2)
                    # print("read y2:", y2)

                    #Make sure we sort our points left to right
                    if x1 > x2:
                        x1,x2 = x2,x1 #neat way to swap in python
                        y1,y2 = y2,y1 

                    #problematic
                    # if scale != 1.0 : #if extend factor is unequal 1.0, we have to calc m/b so that we can extend the line beyond the points given
                    #     l =  np.sqrt(np.square(x1-x2)+np.square(y1-y2)) #calculate length of line
                    #     #print("l:", l ) #debug only
                    #     lextend = (l*scale)/2 # additional length on each side of original line 
                    #     #print("lextend:", lextend) #debug only
                    #     if x1 == x2: #if both X-Values are the same -> perfect vertical line, y=mx+b not valid
                    #         if y1 <= y2: #if y1 is further up
                    #             y1 = y1-int(lextend) #y1 goes further up by lextend
                    #             y2 = y2+int(lextend) #y2 goes further down by lextend 
                    #         else: #otherwise flip it around 
                    #             y1 = y1+int(lextend)
                    #             y2 = y2-int(lextend)

                    #     else:
                    #         m = (y2-y1)/(x2-x1) # m = (y1-y0)/(x1-x0)
                    #         #print("m:", m) #debug only
                    #         b = y1-m*x1 #b = yi-m*xi 
                    #         #print("b:", b) #debug only
                    #         x1 = int(x1 - (lextend / np.sqrt( 1 + np.square(m)) ) ) # x1 - ...,  since left point has to go furhter left
                    #         y1 = int(m*x1 + b)# y = mx+b
                    #         x2 = int(x2 + (lextend / np.sqrt( 1 + np.square(m)) ) ) # x2 + ...,  since left point has to go furhter left
                    #         y2 = int(m*x2 + b)# y = mx+b
                    #         #print("new x1:", x1)#debug only
                    #         #print("new y1:", y1)
                    #         #print("new x2:", x2)
                    #         #print("new y2:", y2)

                    cv.line(img, (x1,y1), (x2,y2), color=color, lineType=linetype, thickness=thickness)       
                 
       # else:
           # print("Can't draw line between given points")
    return

def drawLagLines(cimg, jet_lags, lag_indicies, nozzle_mid_point, midline_coordinates, text=True):
    for i in range(len(jet_lags)):
        if lag_indicies[i] != -1:
            #print(angle_idx[i])
            #print((nozzle_mid_point[0],nozzle_mid_point[1]))
           # cv.line(cimg, (nozzle_mid_point[0][0],nozzle_mid_point[1][0]), (midline_coordinates[1][lag_indicies[i]],midline_coordinates[0][lag_indicies[i]]), color=color_jet_angle, thickness=2, lineType=cv.LINE_AA ) #thickness 2
            cv.line(cimg, (nozzle_mid_point[0][0],nozzle_mid_point[1][0]), (nozzle_mid_point[0][0],midline_coordinates[0][lag_indicies[i]]), color=color_jet_angle, thickness=2, lineType=cv.LINE_AA )
            cv.line(cimg, (nozzle_mid_point[0][0],midline_coordinates[0][lag_indicies[i]]), (midline_coordinates[1][lag_indicies[i]],midline_coordinates[0][lag_indicies[i]]), color=color_jet_angle, thickness=2, lineType=cv.LINE_AA )
            #cv.putText(cimg, "%.2f" %(jet_angles[i]) + "deg", (midline_coordinates[1][angle_indicies[i]],midline_coordinates[0][angle_indicies[i]]), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv.LINE_AA)
            if text:
                #cv.putText(cimg, "%.2f" %(jet_angles[i]) + "deg", (nozzle_mid_point[0][0],midline_coordinates[0][angle_indicies[i]]), cv.FONT_HERSHEY_SIMPLEX, 1, color_text, 2, cv.LINE_AA)
                cv.putText(cimg, "%.1f" %(jet_lags[i]) + " um", (2000 ,40+midline_coordinates[0][lag_indicies[i]]), cv.FONT_HERSHEY_SIMPLEX, 1, color_text, 2, cv.LINE_AA)

    # cv.line(cimg, (nozzle_mid_point[0][0],nozzle_mid_point[1][0]), (int(leftidx_rightidx_dist_midx_midy[-1,3]),int(leftidx_rightidx_dist_midx_midy[-1,4])), color=col, thickness=2, lineType=cv.LINE_AA )
    # cv.line(cimg, (nozzle_mid_point[0][0],nozzle_mid_point[1][0]), (nozzle_mid_point[0][0],int(leftidx_rightidx_dist_midx_midy[-1,4])), color=col, thickness=2, lineType=cv.LINE_AA )
    # cv.line(cimg, (nozzle_mid_point[0][0],int(leftidx_rightidx_dist_midx_midy[-1,4])), (int(leftidx_rightidx_dist_midx_midy[-1,3]),int(leftidx_rightidx_dist_midx_midy[-1,4])), color=col, thickness=2, lineType=cv.LINE_AA )
    return


def drawAngleLines(cimg, jet_angles, angle_indicies, nozzle_mid_point, midline_coordinates, text=True):
    for i in range(len(jet_angles)):
        if angle_indicies[i] != -1:
            #print(angle_idx[i])
            #print((nozzle_mid_point[0],nozzle_mid_point[1]))
            cv.line(cimg, (nozzle_mid_point[0][0],nozzle_mid_point[1][0]), (midline_coordinates[1][angle_indicies[i]],midline_coordinates[0][angle_indicies[i]]), color=color_jet_angle, thickness=2, lineType=cv.LINE_AA ) #thickness 2
            cv.line(cimg, (nozzle_mid_point[0][0],nozzle_mid_point[1][0]), (nozzle_mid_point[0][0],midline_coordinates[0][angle_indicies[i]]), color=color_jet_angle, thickness=2, lineType=cv.LINE_AA )
            cv.line(cimg, (nozzle_mid_point[0][0],midline_coordinates[0][angle_indicies[i]]), (midline_coordinates[1][angle_indicies[i]],midline_coordinates[0][angle_indicies[i]]), color=color_jet_angle, thickness=2, lineType=cv.LINE_AA )
            #cv.putText(cimg, "%.2f" %(jet_angles[i]) + "deg", (midline_coordinates[1][angle_indicies[i]],midline_coordinates[0][angle_indicies[i]]), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv.LINE_AA)
            if text:
                #cv.putText(cimg, "%.2f" %(jet_angles[i]) + "deg", (nozzle_mid_point[0][0],midline_coordinates[0][angle_indicies[i]]), cv.FONT_HERSHEY_SIMPLEX, 1, color_text, 2, cv.LINE_AA)
                cv.putText(cimg, "%.1f" %(jet_angles[i]) + " deg", (2000 ,midline_coordinates[0][angle_indicies[i]]), cv.FONT_HERSHEY_SIMPLEX, 1, color_text, 2, cv.LINE_AA)

    # cv.line(cimg, (nozzle_mid_point[0][0],nozzle_mid_point[1][0]), (int(leftidx_rightidx_dist_midx_midy[-1,3]),int(leftidx_rightidx_dist_midx_midy[-1,4])), color=col, thickness=2, lineType=cv.LINE_AA )
    # cv.line(cimg, (nozzle_mid_point[0][0],nozzle_mid_point[1][0]), (nozzle_mid_point[0][0],int(leftidx_rightidx_dist_midx_midy[-1,4])), color=col, thickness=2, lineType=cv.LINE_AA )
    # cv.line(cimg, (nozzle_mid_point[0][0],int(leftidx_rightidx_dist_midx_midy[-1,4])), (int(leftidx_rightidx_dist_midx_midy[-1,3]),int(leftidx_rightidx_dist_midx_midy[-1,4])), color=col, thickness=2, lineType=cv.LINE_AA )
    return

def drawSpinlinelengthPositionMarkers(cimg, lengths, length_indices, midline, text=True):
    #col = (0,242,255)
    col = (0,255,0)
    for i in range(len(length_indices)):
        if lengths[i] != -1:
            cv.drawMarker(cimg, (midline[1][length_indices[i]],midline[0][length_indices[i]]), color=color_jet_length, thickness=5, markerSize=10, markerType=cv.MARKER_DIAMOND)
            if text:
                #cv.putText(cimg, "%.2f" %abs(lengths[i]) + "um", (midline[1][length_indices[i]]+30,midline[0][length_indices[i]]), cv.FONT_HERSHEY_SIMPLEX, 1, color_text, 2, cv.LINE_AA)
                cv.putText(cimg, "%.1f" %abs(lengths[i]) + " um", (2000,midline[0][length_indices[i]]-50), cv.FONT_HERSHEY_SIMPLEX, 1, color_text, 2, cv.LINE_AA)


    return

def drawDiameterLines(cimg, diameters, left_contour, right_contour, contour_intersection_indices, text=True):
    j = 0
    for i in contour_intersection_indices:
        if i[0] != -1:
            cv.line(cimg, (left_contour[1][i[0]], left_contour[0][i[0]]), (right_contour[1][i[1]], right_contour[0][i[1]]), color=color_diameter_lines, thickness=5, lineType=cv.LINE_AA)
            if text:
                #cv.putText(cimg, "%.1f" %abs(diameters[j]) + "um", (right_contour[1][i[1]]+50, right_contour[0][i[1]]), cv.FONT_HERSHEY_SIMPLEX, 1, color_text, 2, cv.LINE_AA)
                if j%2:
                    cv.putText(cimg, "%.1f" %abs(diameters[j]) + "um", (2250, right_contour[0][i[1]]), cv.FONT_HERSHEY_SIMPLEX, 1, color_text, 2, cv.LINE_AA)
                else:
                    cv.putText(cimg, "%.1f" %abs(diameters[j]) + "um", (2100, right_contour[0][i[1]]), cv.FONT_HERSHEY_SIMPLEX, 1, color_text, 2, cv.LINE_AA)

        j += 1
    return

def drawContours(cimg, contours, left_contour_index, right_contour_index):
        cv.drawContours(cimg, contours, left_contour_index, color=(36,177,34), thickness=2, lineType=cv.LINE_AA) #thickness 2
        cv.drawContours(cimg, contours, right_contour_index, color=(39,127,255), thickness=2, lineType=cv.LINE_AA)
        return

def drawJetEdges(cimg, left_edge_coordinates, right_edge_coordinates, left_color=color_left_edge, right_color=color_right_edge, thickness=2, linetype=cv.LINE_AA): #thickness 2
   # cimg[left_edge_coordinates] = (36,177,34)
    #cimg[right_edge_coordinates] = (39,127,255)
    lineBetweenPoints(cimg, left_edge_coordinates, dimx=1, dimy=0, thickness=thickness, color=left_color, linetype=linetype)
    lineBetweenPoints(cimg, right_edge_coordinates, dimx=1, dimy=0, thickness=thickness, color=right_color,linetype=linetype)
    return

def drawArea(cimg, areas, area_indices, left_edge, right_edge, text=True):
    #col = color_area_1
    #cv.line(cimg, (left_edge[1][0],left_edge[0][0]), (right_edge[1][0],right_edge[0][0]), color=col, thickness=8, lineType=cv.LINE_AA )
    for i in range(len(areas)):
        if area_indices[i][0] != -1:
            #print(angle_idx[i])
            #print((nozzle_mid_point[0],nozzle_mid_point[1]))
            cv.line(cimg, (left_edge[1][area_indices[i][0]],left_edge[0][area_indices[i][0]]), (right_edge[1][area_indices[i][1]],right_edge[0][area_indices[i][1]]), color=color_area, thickness=6, lineType=cv.LINE_AA ) #thickness 5
            if text:
                cv.putText(cimg, "%.1f" %abs(areas[i][1]) + " um2", (2000,left_edge[0][area_indices[i][0]]+50), cv.FONT_HERSHEY_SIMPLEX, 1, color_text, 2, cv.LINE_AA)
        
    return

def drawVolume(cimg, volumes, volume_indices, left_edge, right_edge, text=True):
    for i in range(len(volumes)):
        if volume_indices[i][0] != -1:
            cv.line(cimg, (left_edge[1][volume_indices[i][0]],left_edge[0][volume_indices[i][0]]), (right_edge[1][volume_indices[i][1]],right_edge[0][volume_indices[i][1]]), color=color_volume, thickness=5, lineType=cv.LINE_AA )
            if text:
                #cv.putText(cimg, "%.1f" %abs(volumes[i][1]) + "um2", (left_edge[1][volume_indices[i][0]]+150,left_edge[0][volume_indices[i][0]]), cv.FONT_HERSHEY_SIMPLEX, 1, color_text, 2, cv.LINE_AA)
                cv.putText(cimg, "%.1f" %abs(volumes[i][1]) + " um3", (2000,left_edge[0][volume_indices[i][0]]), cv.FONT_HERSHEY_SIMPLEX, 1, color_text, 2, cv.LINE_AA)

    return

#Composite functions

def annotateAllContours(cimg, contours):
    ncontours = len(contours)
    for i in range(ncontours):  #go through contours 
        col = (np.flip((colormaps['spring'](np.linspace(0.0, 1.0, ncontours))[np.newaxis, :, :3])[0], axis=1)*255)[i] #divides the spring colormap over the number of contours and gives them a color
        cv.drawContours(cimg, contours, i, thickness=2, color=col) #draw them into the image
    return cimg

def annotateEdes(cimg, shape):
    drawJetEdges(cimg, shape["left_contour"], shape["right_contour"], color_left_edge, color_right_edge, linetype=cv.LINE_AA) #expect left edge 0, right edge 1 
    return cimg

def annotateMidline(cimg, shape):
    lineBetweenPoints(cimg, shape["midline_coordinates"], dimx=1, dimy=0, thickness=2, color=color_midline, linetype=cv.LINE_AA) #thicknes = 2
    return cimg 

#for init function
def annotate_img(cimg=[], contours=[], jet_edges=([],[]), midline=[]):

    #if we got contours, draw them in the image 
    if contours:
        ncontours = len(contours)
        for i in range(ncontours):  #go through contours 
            col = (np.flip((colormaps['spring'](np.linspace(0.0, 1.0, ncontours))[np.newaxis, :, :3])[0], axis=1)*255)[i] #divides the spring colormap over the number of contours and gives them a color
            cv.drawContours(cimg, contours, i, thickness=4, color=col) #draw them into the image

    #if we got left/right edge 
    if jet_edges[0]:
        drawJetEdges(cimg, jet_edges[0], jet_edges[1], color_left_edge, color_right_edge, linetype=cv.LINE_AA) #expect left edge 0, right edge 1 

    #if we got a midline 
    if midline:
        lineBetweenPoints(cimg, midline, dimx=1, dimy=0, thickness=2, color=color_midline, linetype=cv.LINE_AA)

    return cimg

