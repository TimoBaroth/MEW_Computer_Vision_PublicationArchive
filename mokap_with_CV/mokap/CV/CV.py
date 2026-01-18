from mokap.CV import CV_functions as cvt
from mokap.CV import file_handling as fh
from mokap.CV import image_annotation as cva
import numpy as np
import cv2 as cv
from threading import Thread, Event
from typing import NoReturn
import time
from collections import deque
#from datetime import datetime

class CVCalc:
    def __init__(self,
                cvinit='init.pkl',
                minit='measurement.csv',
                mqttwriter='',
                multicam = '',
                camidx=0,
                verbose=False):
        
        self.frametimes = deque(maxlen=20) #queue for frametimes
        self.t = 0
        self.cvinitdict =  fh.loadInitData(cvinit)

        self.minitdict = fh.loadMeasurmentInitCSV(minit)
        self.m_flags = [False,False,False] # List of flags, [0] -> calculate y distance nozzle-midline end, [1] -> calculate lengths along midline, [2] -> calculate diameters along midline
        for item in self.minitdict:
            #print(item)
            if item == "general": #skip general settings 
                continue
            if self.minitdict[item]["mode"] == "AVN" or self.minitdict[item]["mode"] == "RVN": #if we try to take a measurement at rel. or abs. position along y-distance between nozzle and midline end
                self.m_flags[0] = True #set flag that we have to calculate y-distance for each frame
            
            if self.minitdict[item]["mode"] == "AML" or self.minitdict[item]["mode"] =="RML" or item == "jet_length": #if we try to take any measurement at rel./abs. jet length or we want to measure the jet length
                self.m_flags[1] = True #set flag that we have to calculate the jet length along the midline for each frame

            if self.minitdict[item]["mode"] == "FOD" or item == "jet_volume": #if we try to take any measurement at first occurence of a jet dia, or we want to calculate jet volumes
                self.m_flags[2] = True #set flag that we have to calculate diameters along the midline for each frame 
        
        self.mqttwriter = mqttwriter 
        self._mqtt_out: bool = False
        if self.mqttwriter:
            self._mqtt_out = True
            print("[INFO] CV Results will be output to MQTT")

        self.mc = multicam
        self.camidx = camidx
        if self.mc:
            self.CV_raw_lock = self.mc._l_CV_buffers_lock[self.camidx] #get lock 
        else:
            print("[WARNING] No multicam object - can't run CV calculations")

        self.verbose = verbose
        self.annotations = {'All contours': 'c',
                            'Jet midline': 'm',
                            'Jet edges': 'e',
                            'Jet lag': 'jL',
                            'Jet angle': 'ja',
                            'Jet length': 'jl',
                            'Jet diameter': 'jd',
                            'Jet Area': 'jA',
                            'Jet Volume': 'jv',
                            'Collector cutoff': 'l'}
        self.annotate = []
        self.runCVCalc: bool = True
        self.text: bool = False

        self._threads = []
        self.out_img = np.zeros((1000, 1000, 3), dtype=np.uint8) #dummy init

    def extractShape(self, img):
        shape = {}
        #get the jet contours for the current frame                                                                                           
        _, contour_status, left_contour_smoothed, right_contour_smoothed, left_contour, right_contour, _, _, _, _, all_contours, _, y_min_l, y_min_r = cvt.getJetContoursHorizontalLineScan(img, collector_left=self.cvinitdict["collector_left"], collector_right=self.cvinitdict["collector_right"], nozzleleftlowy=self.cvinitdict["left_nozzle_line_points"][1][1], nozzlerightlowy=self.cvinitdict["right_nozzle_line_points"][1][1], nozzleleftlowx=self.cvinitdict["left_nozzle_line_points"][0][1], nozzlerightlowx=self.cvinitdict["right_nozzle_line_points"][0][1], Threshold_kernel=self.cvinitdict["thres_kernel"], Blur_kernel=self.cvinitdict["blur_kernel"], upper_cut_off=self.cvinitdict["upper_cut_off"], lower_cut_off=self.cvinitdict["lower_cut_off"]) 
        if contour_status == "Ok":
            #get the midline coordinates from the smoothed contour coordinates
            midline_coordinates, midline_coordinates_smoothed, midline_status = cvt.getMidlinefromContours(left_contour_smoothed, right_contour_smoothed)
            
        shape_status = (contour_status=="Ok" and midline_status=="Ok")
        if shape_status:
            shape = {
                "left_contour": left_contour,
                "right_contour": right_contour,
                "left_contour_smoothed": left_contour_smoothed,
                "right_contour_smoothed": right_contour_smoothed,
                "midline_coordinates": midline_coordinates,
                "midline_coordinates_smoothed": midline_coordinates_smoothed,
            }

        if self.verbose:
            print("Contour status:", contour_status)
            print("Midline status:", contour_status)
        return shape, shape_status, all_contours, y_min_l, y_min_r


    def takeMeasurements(self,measurement_flags, shape, init_dict, measurement_init_dict, verbose):

        def getMeasurementPositionMidlineIndices(mode, positions):
            # directly using some variables from takeMeasurement scope to streamline function call 
            midline_idx = []
            #print("mode:", mode)
            #print("positions:", positions)
            
            if mode == "AVN" or mode == "RVN": # If we want to measure at rel./abs. vertical distance from nozzle 
                #print("len midline arr", len(shape["midline_coordinates"][0]))
                for i in range(len(positions)):
                    if mode == "AVN":
                        y_search = np.rint(positions[i]/measurement_init_dict["general"]["pixel_size"]) #if we give absolute distance position in um 
                        #print(y_search)
                    else:
                        y_search = np.rint((positions[i]/100) * y_d) #if we give relative position in % (of vertical distance between nozzle and midline end)
                
                    idx = np.where(shape["midline_coordinates"][0] == (y_search + init_dict["nozzle_mid_point"][1][0]))

                    if len(idx[0]) == 0: #if this position is not in our list
                        midline_idx.append(-1) # -1 to signal that there is no valid midline index
                    elif len(idx[0]) > 1: #more than one entry, can happen if mid-line runs horizontal for a short distance => staircase
                        midline_idx.append(int(np.rint(np.mean(idx[0])))) #take middle point of horizontal section, round to next integer
                    else: #if only one entry in list, take that one 
                        midline_idx.append(idx[0][0])

            elif mode == "AML" or mode == "RML":
                for i in range(len(positions)):
                    if mode == "AML":
                        idx = np.where(all_midline_lengths >= positions[i]-0.05) # absolute distance -> search for this distance,
                    else:
                        idx = np.where(all_midline_lengths >= (((all_midline_lengths[-1]/100) * positions[i]))-0.05) # last length = total length-> div 100 and * percentag -> search this length , -0.05 to avoid rounding search error

                    if len(idx[0]) == 0:
                        midline_idx.append(-1) # -1 to signal that there is no valid midline index
                    else:
                        midline_idx.append(idx[0][0]+1)

            elif mode == "FOD":
                for i in range(len(positions)):
                    fd = np.array(all_fibre_diameters) * measurement_init_dict["general"]["pixel_size"] #convert to np array to make np.where work, and scale with pixel size
                    dia_idx = np.where(fd <= positions[i])[0] #take first index where diameter is equal or bigger (as we may not exactly match) to the one we search
                    if np.size(dia_idx) == 0: #if we did not find the corresponding index for the position, we get an empty array
                        midline_idx.append(-1) #then save -1 for the midline index to signal invalid index
                    else: 
                        midline_idx.append(all_dia_midline_intersection_indices[dia_idx[0]]) #lookup corresponding midline index from array 
            #print(midline_idx)
            return midline_idx
                
        
        #create measurement dict, save image number, id, and shape file name
        measurement_dict = {}
        #check flags and do pre-calc , see flag creation for reference 
        if measurement_flags[0]: 
            y_d =(np.max(shape["midline_coordinates"][0]) - np.min(shape["midline_coordinates"][0])) 
            #print(y_d)
        if measurement_flags[1]:
            all_midline_lengths, midline_lengths_status = cvt.getLengthAlongMidline(shape["midline_coordinates"], pixel_size=measurement_init_dict["general"]["pixel_size"])
            #print(type(all_midline_lengths))
        if measurement_flags[2]:
            all_fibre_diameters, all_dia_contour_intersection_indices, all_dia_midline_intersection_indices, all_dia_status = cvt.getFibreDiameterAlongMidline(init_dict["nozzle_mid_point"], shape["midline_coordinates"], shape["left_contour"], shape["right_contour"])
            #print(all_fibre_diameters)

        #extract the measurements and append to output if we want to save 
        if "jet_lag" in measurement_init_dict:
            lag_idx = getMeasurementPositionMidlineIndices(measurement_init_dict["jet_lag"]["mode"], measurement_init_dict["jet_lag"]["positions"])
            jet_lags, lag_status = cvt.getJetLag(init_dict["nozzle_mid_point"], shape["midline_coordinates"], lag_idx, measurement_init_dict["general"]["pixel_size"])
            measurement_dict["jet_lag"] = {}
            measurement_dict["jet_lag"]["lag_distances"] = jet_lags
            measurement_dict["jet_lag"]["indices"] = lag_idx
            if verbose:
                print("jet lag calc. status:", lag_status)
                print("jet lag distances:", jet_lags)


        if "jet_angle" in measurement_init_dict:
            angle_idx = getMeasurementPositionMidlineIndices(measurement_init_dict["jet_angle"]["mode"], measurement_init_dict["jet_angle"]["positions"])
            #print(angle_idx)
            jet_angles, angle_status = cvt.getJetAngle(init_dict["nozzle_mid_point"], shape["midline_coordinates"], angle_idx)
            measurement_dict["jet_angle"] = {}
            measurement_dict["jet_angle"]["angles"] = jet_angles
            measurement_dict["jet_angle"]["indices"] = angle_idx
            if verbose:
                print("jet angle calc. status:", angle_status)
                print("jet angles:", jet_angles)

        if "jet_diameter" in measurement_init_dict:
            dia_idx = getMeasurementPositionMidlineIndices(measurement_init_dict["jet_diameter"]["mode"], measurement_init_dict["jet_diameter"]["positions"])
            jet_diameters, last_diameter, contour_dia_indices, dia_status = cvt.getFibreDiameterIntersect(init_dict["nozzle_mid_point"], shape["midline_coordinates"], shape["left_contour"], shape["right_contour"], dia_idx, measurement_init_dict["general"]["pixel_size"], measurement_init_dict["general"]["diameter_approx_distance"])

            measurement_dict["jet_diameter"] = {}
            measurement_dict["jet_diameter"]["diameters"] = jet_diameters
           # measurement_dict["jet_diameter"]["last_diameter"] = last_diameter
            measurement_dict["jet_diameter"]["contour_indices"] = contour_dia_indices
            measurement_dict["jet_diameter"]["midline_indices"] = dia_idx
            if verbose:
                print("jet diameter calc. status:", dia_status)
                print("jet diameters:", jet_diameters)
                #print("last diameter:", last_diameter)

        if "jet_length" in measurement_init_dict:
            length_idx = getMeasurementPositionMidlineIndices(measurement_init_dict["jet_length"]["mode"], measurement_init_dict["jet_length"]["positions"])
            jet_lengths = []
            for i in length_idx:
                if i == -1 or i == 0:
                    jet_lengths.append(-1) #invalid
                else:
                    jet_lengths.append(all_midline_lengths[i-1])

            measurement_dict["jet_length"] = {}
            measurement_dict["jet_length"]["lengths"] = jet_lengths
            measurement_dict["jet_length"]["total_length"] = all_midline_lengths[-1]
            measurement_dict["jet_length"]["indices"] = length_idx
            if verbose:
                print("jet length calc. status:", midline_lengths_status)
                print("jet lengths:", jet_lengths)
                print("total length:", all_midline_lengths[-1])

        if "jet_area" in measurement_init_dict:
            area_idx = getMeasurementPositionMidlineIndices(measurement_init_dict["jet_area"]["mode"], measurement_init_dict["jet_area"]["positions"])
            jet_areas, total_area, area_edge_indices, area_status = cvt.getSpinlineArea2_skimage_V4(shape["midline_coordinates"], shape["left_contour"], shape["right_contour"], area_idx, measurement_init_dict["general"]["pixel_size"], 25)
            
            measurement_dict["jet_area"] = {}
            measurement_dict["jet_area"]["areas"] = jet_areas
            measurement_dict["jet_area"]["total_area"] = total_area
            measurement_dict["jet_area"]["contour_indices"] = area_edge_indices
            measurement_dict["jet_area"]["midline_indices"] = area_idx
        
            if verbose:
                print("jet area calc. status:", area_status)
                print("jet areas:", jet_areas)
                print("total area:", total_area)

        if "jet_volume" in measurement_init_dict:
            vol_status = all_dia_status

            if all_dia_status == "Ok":
                volume_idx = getMeasurementPositionMidlineIndices(measurement_init_dict["jet_volume"]["mode"], measurement_init_dict["jet_volume"]["positions"])
                all_volumes, total_volume, volume_status = cvt.getSpinlineVolumes(shape["left_contour"], shape["right_contour"], all_fibre_diameters, all_dia_contour_intersection_indices, measurement_init_dict["general"]["pixel_size"])
        
                jet_volumes = [[-1,-1] for i in range(len(volume_idx))] #list of areas for given area positions
                volume_edge_indices = [[-1,-1] for i in range(len(volume_idx))] 
                volume_midline_indices = [[-1] for i in range(len(volume_idx))]
                vol_status = volume_status 
                if volume_status == "Ok":
                    volume_idx_sorted = sorted(volume_idx) #sort, to make sure calculations are in right order =-> top to bottom of jet, see cvt.getSpinlineArea for reference
                    volume_idx_id_t = [volume_idx.index(i) for i in volume_idx_sorted]
                    volume_idx_id = volume_idx_id_t
                    for i in range(1,len(volume_idx_id_t)):
                        if volume_idx_id_t[i-1] == volume_idx_id_t[i]:
                            volume_idx_id[i] += 1

                    k = 0 
                    lastidx = 0
                    for i in volume_idx_sorted:
                        idx = np.argmin(np.abs(np.array(all_dia_midline_intersection_indices)-i)) #get closest index to match volume_idx with volume sections
                        if idx == 0:
                            jet_volumes[volume_idx_id[k]] = [k, -1] #invalid 
                            volume_midline_indices[volume_idx_id[k]] = -1
                            volume_edge_indices[volume_idx_id[k]] = [-1,-1]
                        else:
                            volume_midline_indices[volume_idx_id[k]] = idx
                            volume_edge_indices[volume_idx_id[k]] = all_dia_contour_intersection_indices[idx]
                            jet_volumes[volume_idx_id[k]] = [k,(sum([all_volumes[j] for j in range(lastidx,idx,1)]))] #calculate jet volume up to index, starting from 0 or last index
                        lastidx = idx 
                        k += 1
                    
            else:
                jet_volumes = [[-1,-1]]
                volume_edge_indices = [[-1,-1]]
                volume_midline_indices = [-1]
                total_volume = -1

            #print(volume_edge_indices)
            #print(volume_midline_indices)
            measurement_dict["jet_volume"] = {}
            measurement_dict["jet_volume"]["volumes"] = jet_volumes
            measurement_dict["jet_volume"]["total_volume"] = total_volume
            measurement_dict["jet_volume"]["contour_indices"] = volume_edge_indices
            measurement_dict["jet_volume"]["midline_indices"] = volume_midline_indices
            if verbose:
                print("jet volume calc. status:", vol_status)
                print("jet volumes:", jet_volumes)
                print("total volume:", total_volume)

       
        return measurement_dict

    def annotateImg(self, cimg, shape_status, all_contours, shape, measurement_dict, y_min_l=0, y_min_r=0):
        
        if self.annotate and shape_status:
            if "c" in self.annotate and all_contours:
                cimg = cva.annotateAllContours(cimg, all_contours)
            if "e" in self.annotate:
                cimg = cva.annotateEdes(cimg, shape)
            if "m" in self.annotate:
                cimg = cva.annotateMidline(cimg, shape)
            if "jL" in self.annotate and "jet_lag" in measurement_dict:
                cva.drawLagLines(cimg, measurement_dict["jet_lag"]["lag_distances"], measurement_dict["jet_lag"]["indices"], self.cvinitdict["nozzle_mid_point"], shape["midline_coordinates"],self.text)
            if "ja" in self.annotate and "jet_angle" in measurement_dict:
                cva.drawAngleLines(cimg, measurement_dict["jet_angle"]["angles"], measurement_dict["jet_angle"]["indices"], self.cvinitdict["nozzle_mid_point"], shape["midline_coordinates"],self.text)
            if "jd" in self.annotate and "jet_diameter" in measurement_dict:
                cva.drawDiameterLines(cimg, measurement_dict["jet_diameter"]["diameters"], shape["left_contour"], shape["right_contour"], measurement_dict["jet_diameter"]["contour_indices"],self.text)
            if "jl" in self.annotate and "jet_length" in measurement_dict:
                cva.drawSpinlinelengthPositionMarkers(cimg, measurement_dict["jet_length"]["lengths"], measurement_dict["jet_length"]["indices"], shape["midline_coordinates"],self.text)
            if "jA" in self.annotate and "jet_area" in measurement_dict:
                cva.drawArea(cimg, measurement_dict["jet_area"]["areas"], measurement_dict["jet_area"]["contour_indices"], shape["left_contour"], shape["right_contour"],self.text)
            if "jv" in self.annotate and "jet_volume" in measurement_dict:
                cva.drawVolume(cimg,  measurement_dict["jet_volume"]["volumes"], measurement_dict["jet_volume"]["contour_indices"], shape["left_contour"], shape["right_contour"],self.text)
            if "l" in self.annotate and y_min_l != 0 and y_min_r != 0:
                if y_min_r == y_min_l:
                    cva.lineBetweenPoints(cimg,([0,cimg.shape[1]],[y_min_l,y_min_r])) 
                else:
                    cva.lineBetweenPoints(cimg,([0,cimg.shape[1]],[y_min_l,y_min_r]),color=(255,0,255)) 

        return cimg
    

    def _cv_thread(self)-> NoReturn:
        # print("[INFO] Start CV thread")
        # csv_file_path = "E:/" + "cv_frames" +str(datetime.now()).replace(" ","_") +".csv"
        # csv_file = open(csv_file_path, "w", newline='')
        # csv_writer = csv.writer(csv_file, dialect='excel')
        # header = ['Imag']
        # for item in self.mqttlogger.values:
        #     header.append(str(item))
        # csv_writer.writerow(header)
        while self.runCVCalc:
            self.mc._l_CV_buffers_event[self.camidx].wait() #wait until we got a frame to process
            self.mc._l_CV_buffers_lock[self.camidx].acquire() #get the lock so, so we can work on the frame without it being manipulated by othe threads
            img = self.mc._l_CV_buffers[self.camidx] #the image we are working on
            shape, shape_status, all_contours, y_min_l, y_min_r = self.extractShape(img) # try to extract jet contour and midline 
            if shape_status: #if ok
                measurement_dict = self.takeMeasurements(self.m_flags, shape, self.cvinitdict, self.minitdict, self.verbose) #take measurements 
                if self._mqtt_out:
                    self.mqttwriter.publish_msg(measurement_dict)
                self.frametimes.append(time.monotonic())
            else:
                measurement_dict = {}

            self.out_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR) #create color img     
            if self.annotate: #creates annotated image
                self.out_img = self.annotateImg(self.out_img, shape_status, all_contours, shape, measurement_dict, y_min_l, y_min_r)
            dtsum = 0
            queuelen = len(self.frametimes)
            for i in range(queuelen-1):
                dtsum += self.frametimes[i+1] - self.frametimes[i]
            if queuelen != 0:
                dtavg = dtsum/queuelen
            else:
                dtavg = dtsum
            avgfps = -1
            if dtavg != 0:
                avgfps = 1/dtavg 
            cv.putText(self.out_img, "Avg. dT over last " + str(queuelen) + " frames: "+ "%.2f" %(dtavg) + " s", (50 ,50), cv.FONT_HERSHEY_SIMPLEX, 1, (100,100,255), 5, cv.LINE_AA)
            cv.putText(self.out_img, "Avg. FPS over last " + str(queuelen) + " frames: "+ "%.1f" %(avgfps), (50 ,100), cv.FONT_HERSHEY_SIMPLEX, 1, (100,100,255), 5, cv.LINE_AA)
            cv.putText(self.out_img, "shape_status" + str(shape_status) , (50 ,150), cv.FONT_HERSHEY_SIMPLEX, 1, (100,100,255), 5, cv.LINE_AA)

            self.mc._l_CV_buffers_lock[self.camidx].release() #release lock 
            self.mc._l_CV_buffers_event[self.camidx].clear()  #reset event 


    def startCV(self):
        g = Thread(target=self._cv_thread, daemon=True) 
        g.start()
        self._threads.append(g)

