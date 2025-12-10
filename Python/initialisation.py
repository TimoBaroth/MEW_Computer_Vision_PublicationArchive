import CV_functions as cvt
import image_annotation as cva
import cv2 as cv 
import pickle


# img_path = r"D:\test\prep\init.jpg"
# output_path = r"D:\test\prep"

def main(img_path, output_path):

    output_file_name = "\CV_init"

    output_file = output_path + output_file_name + ".pkl"

    #img_file = r"C:\Users\n11438177\OneDrive - Queensland University of Technology\Documents\My Pictures\vlcsnap-2024-03-08-18h11m03s438.png"

    img = cv.imread(img_path,cv.IMREAD_GRAYSCALE)
    cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR) #make copy in BGR space so that we can have colored markers, lines, text etc.
    cimg1 = cimg.copy()
    cimg2 = cimg.copy()
    img_shape = img.shape
    img_width = img.shape[1]
    img_height = img.shape[0]


    window_resize_scaler = 2.8 #scaler => e.g. 2 => resized window will have 1/2 size 
    window_size = (int(img_shape[1]/window_resize_scaler), int(img_shape[0]/window_resize_scaler)) #(width, heigth)

    blur_kernel = 0
    thres_kernel = 3
    upper_cut_off = 0
    lower_cut_off = 0
    collector_left = img_height
    collector_right = img_height

    def callback_blurkernel(val): 
        global blur_kernel
        blur_kernel = val
        return

    def callback_threskernel(val): 
        global thres_kernel
        thres_kernel = val
        return

    def callback(val): #dummy callback
        return


    cv.namedWindow('Initialisation_MEW_CV', cv.WINDOW_GUI_EXPANDED)
    cv.resizeWindow("Initialisation_MEW_CV", width=window_size[0], height=window_size[1])  #resize a bit, since our camera makes very large images that don't fit on screen
    cv.createTrackbar('Upper Cutoff', 'Initialisation_MEW_CV', upper_cut_off, img_height, callback) #trackbar, init to left point lower Y coordinate
    cv.createTrackbar('Lower Cutoff', 'Initialisation_MEW_CV', lower_cut_off, img_height, callback) #trackbar, init to left point lower Y coordinate
    cv.createTrackbar('Collector Left', 'Initialisation_MEW_CV', collector_left, img_height, callback) #trackbar, init to left point lower Y coordinate
    cv.createTrackbar('Collector Right', 'Initialisation_MEW_CV', collector_right, img_height, callback) #trackbar, init to left point lower Y coordinate
    cv.createTrackbar('Blur Kernel', 'Initialisation_MEW_CV', 0, 51, callback_blurkernel) #trackbar, 
    cv.setTrackbarMin('Blur Kernel', 'Initialisation_MEW_CV', 0)
    cv.createTrackbar('Threshold Kernel', 'Initialisation_MEW_CV', 3, 201, callback_threskernel) #trackbar, 
    cv.setTrackbarMin('Threshold Kernel', 'Initialisation_MEW_CV', 3)

    cv.namedWindow('Thresholded image', cv.WINDOW_GUI_EXPANDED)
    cv.resizeWindow('Thresholded image', width=window_size[0], height=window_size[1]) 

    cv.namedWindow('CV Result', cv.WINDOW_GUI_EXPANDED)
    cv.resizeWindow('CV Result', width=window_size[0], height=window_size[1]) 

    nozzle_pos_init = False 
    upper_cut_off_init = False
    lower_cut_off_init = False
    collector_init = False
    draw_edges = True
    draw_Contours = False

    while True:

        k = cv.pollKey() & 0xFF #0xFF has something to do with key codes 
        if k == 105: #if "i" is pressed  =>[i]nitialisation
            left_nozzle_line_points, right_nozzle_line_points, nozzle_mid_point = cvt.InitNozzlePosition(img, window_resize_scaler=window_resize_scaler)
            nozzle_x = int(nozzle_mid_point[0][0])
            nozzle_y = int(nozzle_mid_point[1][0])
            nozzle_leftlowx = left_nozzle_line_points[0][1]
            nozzle_leftlowy = left_nozzle_line_points[1][1]
            nozzle_rightlowx = right_nozzle_line_points[0][1]
            nozzle_rightlowy = right_nozzle_line_points[1][1]
            nozzle_pos_init = True
        elif k == 27: #if escape key was pressed
            cv.destroyAllWindows #close windows
            break #break out of while loop
        elif k == 117: #if "u" is pressed and our nozzle position is initialised => [u]pper cut off
            upper_cut_off_init = not(upper_cut_off_init)  #toggle
        elif k == 108: #if "l" is pressed and our nozzle position is initialised => [l]lower cut off
            lower_cut_off_init = not(lower_cut_off_init)  #toggle
        elif k == 115: # if "s" was pressed save our nozzle and cut off position
            if nozzle_pos_init == True and upper_cut_off_init == True and lower_cut_off_init == True and collector_init ==True:
                outdict = {
                    "left_nozzle_line_points": left_nozzle_line_points,
                    "right_nozzle_line_points": right_nozzle_line_points,
                    "nozzle_mid_point": nozzle_mid_point,
                    "upper_cut_off": upper_cut_off,
                    "lower_cut_off": lower_cut_off,
                    "blur_kernel": blur_kernel,
                    "thres_kernel": thres_kernel,
                    "collector_left": collector_left,
                    "collector_right": collector_right

                }
                with open(output_file, 'wb') as fp:
                    pickle.dump(outdict, fp)
                    print("Output file saved")
        elif k == 101: # if "e" was pressed => [e]dges
            draw_edges = not(draw_edges) #toggle
        elif k == 69: #if "E" was pressed => raw [E]dges
            draw_Contours = not(draw_Contours)
        elif k == 99: #if "c" was pressed => raw [c]ollector
            collector_init = not(collector_init)
            
        if k != -1: #for some reason we have to call this again if we had a key press, otherwise window will be resized to original size and not have expanded gui
            cimg1 = cimg.copy() #make copy in BGR space so that we can have colored markers, lines, text etc.
            cv.namedWindow('Initialisation_MEW_CV', cv.WINDOW_GUI_EXPANDED)
            cv.resizeWindow("Initialisation_MEW_CV", width=window_size[0], height=window_size[1])  #resize a bit, since our camera makes very large images that don't fit on screen

        trackbar_kernel = cv.getTrackbarPos('Blur Kernel', 'Initialisation_MEW_CV') 
        if (trackbar_kernel%2)==0: #only odd kernel size is valid
            if blur_kernel != 0:
                blur_kernel = trackbar_kernel+1
            cv.setTrackbarPos("Blur Kernel", "Initialisation_MEW_CV", blur_kernel)
        else:
            blur_kernel=trackbar_kernel
            cv.setTrackbarPos("Blur Kernel", "Initialisation_MEW_CV", blur_kernel)



        trackbar_kernel = cv.getTrackbarPos('Threshold Kernel', 'Initialisation_MEW_CV') 
        if (trackbar_kernel%2)==0: #only odd kernel size is valid
            thres_kernel = trackbar_kernel+1
            cv.setTrackbarPos("Threshold Kernel", "Initialisation_MEW_CV", thres_kernel)
        else:
            thres_kernel=trackbar_kernel
            cv.setTrackbarPos("Threshold Kernel", "Initialisation_MEW_CV", thres_kernel)

        if nozzle_pos_init == True:
            cva.mark(cimg1, left_nozzle_line_points, markersize=10, thickness=3)
            cva.mark(cimg1, right_nozzle_line_points, markersize=10, thickness=3)
            cva.mark(cimg1, nozzle_mid_point, markersize=10, thickness=3)
            cva.lineBetweenPoints(cimg1, left_nozzle_line_points, thickness=3)
            cva.lineBetweenPoints(cimg1, right_nozzle_line_points,thickness=3)
            cva.lineBetweenPoints(cimg1, [[left_nozzle_line_points[0][-1],right_nozzle_line_points[0][-1]],[left_nozzle_line_points[1][-1],right_nozzle_line_points[1][-1]]], thickness=3)

        if upper_cut_off_init == True:
            cv.line(cimg1, (0, upper_cut_off), (img_shape[1], upper_cut_off), color=(0,0,255), thickness=2, lineType=cv.LINE_AA) 
        else:
            upper_cut_off = cv.getTrackbarPos('Upper Cutoff', 'Initialisation_MEW_CV') 
            cv.line(cimg1, (0, upper_cut_off), (img_shape[1], upper_cut_off), color=(0,255,0), thickness=2, lineType=cv.LINE_AA)

        if lower_cut_off_init == True:
            cv.line(cimg1, (0, lower_cut_off), (img_shape[1], lower_cut_off), color=(0,255,255), thickness=2, lineType=cv.LINE_AA)
        else:
            lower_cut_off = cv.getTrackbarPos('Lower Cutoff', 'Initialisation_MEW_CV') 
            cv.line(cimg1, (0, lower_cut_off), (img_shape[1], lower_cut_off), color=(255,255,0), thickness=2, lineType=cv.LINE_AA)

        if collector_init == True:
            cv.line(cimg1, (0, collector_left), (img_shape[1], collector_right), color=(255,0,0), thickness=2, lineType=cv.LINE_AA)
        else:
            collector_left = cv.getTrackbarPos('Collector Left', 'Initialisation_MEW_CV') 
            collector_right = cv.getTrackbarPos('Collector Right', 'Initialisation_MEW_CV') 
            cv.line(cimg1, (0, collector_left), (img_shape[1], collector_right), color=(255,0,255), thickness=2, lineType=cv.LINE_AA)


        # if blur_kernel != 0:
        #     img2 = cv.GaussianBlur(img,(blur_kernel,blur_kernel),0)
        #     thres_img = cv.adaptiveThreshold(img2,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,thres_kernel,2) 
        # else:
        #     thres_img = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,thres_kernel,2) 
        # cv.imshow('Thresholded image', thres_img) #show the pictures
    

        if nozzle_pos_init == True and upper_cut_off_init == True and lower_cut_off_init == True and collector_init ==True: 
            #print("all init, try CV")
            try:
                adgimage, contour_status, left_contour_smoothed, right_contour_smoothed, left_edge, right_edge, ll_limit, lr_limit, rl_limit, rr_limit, all_contours, contours, y_min_l, y_min_r = cvt.getJetContoursHorizontalLineScan(img, collector_left=collector_left, collector_right=collector_right, nozzleleftlowy=nozzle_leftlowy, nozzlerightlowy=nozzle_rightlowy, nozzleleftlowx=nozzle_leftlowx, nozzlerightlowx=nozzle_rightlowx, Threshold_kernel=thres_kernel, Blur_kernel=blur_kernel, upper_cut_off=upper_cut_off, lower_cut_off=lower_cut_off) 
                cv.imshow('Thresholded image', adgimage) #show the pictures
                print(contour_status)
                if  contour_status == "Ok":
                    cimg2 = cimg.copy()
                    midline_coordinates, midline_coordinates_smoothed, midline_status = cvt.getMidlinefromContours(left_contour_smoothed, right_contour_smoothed)
                    cimg2 = cva.annotate_img(cimg2, contours=all_contours, jet_edges=(left_edge,right_edge), midline=midline_coordinates)
                    cv.imshow('CV Result', cimg2) #show the pictures

            except:
                pass
            

        cv.imshow('Initialisation_MEW_CV', cimg1) #show the pictures #
    return