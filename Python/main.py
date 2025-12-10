import CV_functions as cvt
import image_annotation as cva
import initialisation as cvinit 
import file_handling as fh
import cv2 as cv 
import pickle

import csv
import os
import lzma
import numpy as np
import time
import os
import sys, getopt

def main(argv):

    # a few helper functions to avoid code duplication
    def exitGracefully():
        nonlocal shape_out_csv, measurement_out_csv #so we can just call it and don't have to remember to give it the csv files
        #check if we have open files, if so close them before we exit
        if shape_out_csv:
            shape_out_csv.close()
        if measurement_out_csv:
            measurement_out_csv.close()
        
        sys.exit()
        return

    def loadInit(init_path):
        init_dict = fh.loadInitData(init_path)
        if not init_dict:
            print("Unable to load initialisation data")
            exitGracefully() #if we failed to load it exit
        return init_dict
    
    def listImgs(img_path):
        img_dict, img_keys, img_count = fh.listImagesNEW(img_path)
        if not img_dict:
            print("Unable to list files from image folder path")
            exitGracefully() #if we failed to load it exit
        else:
            print("Image folder files listed. Total number of files:", img_count)
        return img_dict, img_keys, img_count 
    
    def createShapeOut(out_path):
        shape_out_path = fh.createSubDir(out_path,"Shapes")
        if not shape_out_path:
            print("Unable to create sub-directory for shapes")
            exitGracefully()

        shape_out_csv, shape_out_csv_writer = fh.createCSV(out_path, "Shapes")
        if not shape_out_csv or not shape_out_csv_writer:
            print("Unable to create CSV file for shapes or file already exists")
            exitGracefully()
        return shape_out_path, shape_out_csv, shape_out_csv_writer
    
    def getIterator(count):
        #set up some variables for our loop over the images/shapes that we want to analyse 
        iterator = start_idx-1  #variable to iterate over in our loop, init with where we want to start processing 
        if stop_idx == -1: #calc the number of images/shapes we have to analyse 
            number_of_steps = count - (start_idx-1)
        else:
            number_of_steps = stop_idx - (start_idx-1)
        return iterator, number_of_steps

    def loadImg(img_key):
        img = []
        image_loaded = False
        if not (img_key in img_dict): #check if we have the requested image number in our dict/list..
            print("Can't load image with number:", img_key, " This image is not available.")
        else:
            #load image
            imgpath = os.path.join(img_path, img_dict[img_key][0])
            #imgpath = img_folder_path + "\\" + img_dict[img_key][0]
            img = cv.imread(imgpath,cv.IMREAD_GRAYSCALE) #load it
            image_loaded = True
        return img, image_loaded

    def extractShape(img, init_dict):
        shape = {}
        #get the jet contours for the current frame                                                                                           
        thres_img, contour_status, left_contour_smoothed, right_contour_smoothed, left_contour, right_contour, _, _, _, _, all_contours, _, y_min_l, y_min_r = cvt.getJetContoursHorizontalLineScan(img, collector_left=init_dict["collector_left"], collector_right=init_dict["collector_right"], nozzleleftlowy=init_dict["left_nozzle_line_points"][1][1], nozzlerightlowy=init_dict["right_nozzle_line_points"][1][1], nozzleleftlowx=init_dict["left_nozzle_line_points"][0][1], nozzlerightlowx=init_dict["right_nozzle_line_points"][0][1], Threshold_kernel=init_dict["thres_kernel"], Blur_kernel=init_dict["blur_kernel"], upper_cut_off=init_dict["upper_cut_off"], lower_cut_off=init_dict["lower_cut_off"]) 
        if verbose:
            print("Contour status:", contour_status)
        if contour_status == "Ok":
            #get the midline coordinates from the smoothed contour coordinates
            midline_coordinates, midline_coordinates_smoothed, midline_status = cvt.getMidlinefromContours(left_contour_smoothed, right_contour_smoothed)
            if verbose:
                print("Midline status:", midline_status)


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
        return thres_img, shape, shape_status, all_contours, y_min_l, y_min_r


    def saveShapeNEW(shape, img_no, shape_out_path, shape_out_csv_writer, verbose):
        if shape:
            output_file_name = "CV_shape_" + "image_"+str(img_no)+".xz"
            output_file_path = os.path.join(shape_out_path, output_file_name)
            #lzma compression gives muche smaler files , approx factor 3 compared to pickle without lzma
            with lzma.open(output_file_path, 'wb') as fp:
                pickle.dump(shape, fp)
                if verbose:
                    print("Shape for image No.:", img_no, " saved.")
        else:
            output_file_name = "" #set empty if we have not saved an output file
            if verbose:
                print("Ivalid shape extraction for image No.:", img_no, " no output file saved.")
            
        csvline = [str(img_no), output_file_name] #create csv file line
        shape_out_csv_writer.writerow(csvline) #write it to csv file
        return output_file_name


    def saveShape(shape, img_no, img_id, shape_out_path, shape_out_csv_writer, verbose):
        if shape:
            output_file_name = "CV_shape_" + "image_"+str(img_key)+"_id_"+str(img_dict[img_key][1])+".xz"
            output_file_path = os.path.join(shape_out_path, output_file_name)
            #lzma compression gives muche smaler files , approx factor 3 compared to pickle without lzma
            with lzma.open(output_file_path, 'wb') as fp:
                pickle.dump(shape, fp)
                if verbose:
                    print("Shape for image No.:", img_no, " saved.")
        else:
            output_file_name = "" #set empty if we have not saved an output file
            if verbose:
                print("Ivalid shape extraction for image No.:", img_no, " no output file saved.")
            
        csvline = [str(img_no), str(img_id), output_file_name] #create csv file line
        shape_out_csv_writer.writerow(csvline) #write it to csv file
        return output_file_name

    def loadMeasurementInit(path):
        measurement_init_dict = fh.loadMeasurmentInitCSV(path)
        if not measurement_init_dict:
            print("Unable to load measurement initialisation data")
            exitGracefully() #if we failed to load it exit
        return measurement_init_dict

    def createMeasurementOut(out_path):
        #measurement_out_path = fh.createSubDir(out_path,"Measurements")
        #if not measurement_out_path:
        #    print("Unable to create sub-directory for measurements")
        #    exitGracefully()
        measurement_out_csv, measurement_out_csv_writer = fh.createCSV(out_path, "Measurements")
        if not measurement_out_csv or not measurement_out_csv_writer:
            print("Unable to create CSV file for measurements or file already exists")
            exitGracefully()
       #return measurement_out_path, measurement_out_csv, measurement_out_csv_writer
        return measurement_out_csv, measurement_out_csv_writer

    def setupMeasurementsHeader(measurement_init_dict):
        if "jet_lag" in measurement_init_dict: #check if we have entry with name "jet_angle"
            measurement_csv_header.append("JetLag")

        if "jet_angle" in measurement_init_dict: #check if we have entry with name "jet_angle"
            measurement_csv_header.append("JetAngle")
           # measurement_csv_header.append("last_jet_angle")

        if "jet_diameter" in measurement_init_dict: #check if we have entry with name "diameter"
            measurement_csv_header.append("JetDiameter")
           # measurement_csv_header.append("last_fiber_diameter")

        if "jet_length" in measurement_init_dict: #check if we have entry with name "length"
            measurement_csv_header.append("JetLength")
            measurement_csv_header.append("total_jet_length")

        if "jet_area" in measurement_init_dict: #check if we have entry with name "area"
            measurement_csv_header.append("JetArea")
            measurement_csv_header.append("total_jet_area")

        if "jet_volume" in measurement_init_dict: #check if we have entry with name "volume"
            measurement_csv_header.append("JetVolume")
        return measurement_csv_header

    def getMeasurementReferenceFlags(measurement_init_dict):
        flags = [False,False,False] # List of flags, [0] -> calculate y distance nozzle-midline end, [1] -> calculate lengths along midline, [2] -> calculate diameters along midline
        for item in measurement_init_dict:
            #print(item)
            if item == "general": #skip general settings 
                continue
           # print(measurement_init_dict[item]["mode"])

            if measurement_init_dict[item]["mode"] == "AVN" or measurement_init_dict[item]["mode"] == "RVN": #if we try to take a measurement at rel. or abs. position along y-distance between nozzle and midline end
                flags[0] = True #set flag that we have to calculate y-distance for each frame
            
            if measurement_init_dict[item]["mode"] == "AML" or measurement_init_dict[item]["mode"] =="RML" or item == "jet_length": #if we try to take any measurement at rel./abs. jet length or we want to measure the jet length
                flags[1] = True #set flag that we have to calculate the jet length along the midline for each frame

            if measurement_init_dict[item]["mode"] == "FOD" or item == "jet_volume": #if we try to take any measurement at first occurence of a jet dia, or we want to calculate jet volumes
                flags[2] = True #set flag that we have to calculate diameters along the midline for each frame 

        return flags

    def listShapes(shape_path):
        shape_dict, shape_keys, shape_count = fh.listShapesNEW(shape_path)
        if not shape_dict:
            print("Unable to list shapes from shape folder path")
            exitGracefully() #if we failed to load it exit
        else:
            print("Shape folder files listed. Total number of shapes:", shape_count)
        return shape_dict, shape_keys, shape_count

    def loadShape(shape_key):
        shape = {}
        try:
            shapepath = os.path.join(shape_path, shape_dict[shape_key][0])
            with lzma.open(shapepath) as f:
                shape = pickle.load(f)
        except:
            print("Error loading shape for image no.:", shape_key)
        return shape
    
    def takeMeasurements(measurement_flags, shape, img_no, shape_file, init_dict, measurement_init_dict, verbose):

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
        measurement_dict["general"] = {}
        measurement_dict["general"]["img_No"] = img_no
        measurement_dict["general"]["shape_file"] = shape_file
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

        if "jet_lag" in measurement_init_dict:
            lag_idx = getMeasurementPositionMidlineIndices(measurement_init_dict["jet_lag"]["mode"], measurement_init_dict["jet_lag"]["positions"])
            jet_lags, lag_status = cvt.getJetLag(init_dict["nozzle_mid_point"], shape["midline_coordinates"], lag_idx, measurement_init_dict["general"]["pixel_size"])
            measurement_dict["jet_lag"] = {}
            measurement_dict["jet_lag"]["lag_distances"] = jet_lags
            measurement_dict["jet_lag"]["indices"] = lag_idx
            if verbose:
                print("jet lag calc. status:", lag_status)
                print("jet lag distances:", jet_lags)


        #extract the measurements and append to output if we want to save 
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

    def saveMeasurements(measurement_dict, csv_writer, verbose):
        measurement_result_row = []
        measurement_result_row.append(str(measurement_dict["general"]["img_No"]))
        #measurement_result_row.append(str(measurement_dict["general"]["shape_file"]))

        #output_file_name = "CV_measurement_" + "image_"+str(measurement_dict["general"]["img_No"])+".xz" 
        #output_file_path = os.path.join(out_path, output_file_name)

        #measurement_result_row.append(str(output_file_name))

        if "jet_lag" in measurement_dict:
            measurement_result_row.append(str(measurement_dict["jet_lag"]["lag_distances"]))
        if "jet_angle" in measurement_dict:
            measurement_result_row.append(str(measurement_dict["jet_angle"]["angles"]))
            #measurement_result_row.append(str(measurement_dict["jet_angle"]["last_angle"]))
        if "jet_diameter" in measurement_dict:
            measurement_result_row.append(str(measurement_dict["jet_diameter"]["diameters"]))
           # measurement_result_row.append(str(measurement_dict["jet_diameter"]["last_diameter"]))
        if "jet_length" in measurement_dict:
            measurement_result_row.append(str(measurement_dict["jet_length"]["lengths"] ))
            measurement_result_row.append(str(measurement_dict["jet_length"]["total_length"]))
        if "jet_area" in measurement_dict:
            measurement_result_row.append(str(measurement_dict["jet_area"]["areas"]))
            measurement_result_row.append(str(measurement_dict["jet_area"]["total_area"]))
        if "jet_volume" in measurement_dict:
            measurement_result_row.append(str(measurement_dict["jet_volume"]["volumes"]))
            
        #lzma compression gives muche smaler files , approx factor 3 compared to pickle without lzma
        #with lzma.open(output_file_path, 'wb') as fp:
            #pickle.dump(shape, fp)
              
        csv_writer.writerow(measurement_result_row)

        if verbose:
            print("Measurement for image No.:", str(measurement_dict["general"]["img_No"]), " saved.")

        return

    def annotateImg(cimg, annotate, shape_status, all_contours, shape, measurement_dict, init_dict, text, y_min_l=0, y_min_r=0):
        if annotate and shape_status:
            if "c" in annotate and all_contours:
                cimg = cva.annotateAllContours(cimg, all_contours)
            if "e" in annotate:
                cimg = cva.annotateEdes(cimg, shape)
            if "m" in annotate:
                cimg = cva.annotateMidline(cimg, shape)
            if "jL" in annotate and "jet_lag" in measurement_dict:
                cva.drawLagLines(cimg, measurement_dict["jet_lag"]["lag_distances"], measurement_dict["jet_lag"]["indices"], init_dict["nozzle_mid_point"], shape["midline_coordinates"],text)
            if "ja" in annotate and "jet_angle" in measurement_dict:
                cva.drawAngleLines(cimg, measurement_dict["jet_angle"]["angles"], measurement_dict["jet_angle"]["indices"], init_dict["nozzle_mid_point"], shape["midline_coordinates"],text)
            if "jd" in annotate and "jet_diameter" in measurement_dict:
                cva.drawDiameterLines(cimg, measurement_dict["jet_diameter"]["diameters"], shape["left_contour"], shape["right_contour"], measurement_dict["jet_diameter"]["contour_indices"],text)
            if "jl" in annotate and "jet_length" in measurement_dict:
                cva.drawSpinlinelengthPositionMarkers(cimg, measurement_dict["jet_length"]["lengths"], measurement_dict["jet_length"]["indices"], shape["midline_coordinates"],text)
            if "jA" in annotate and "jet_area" in measurement_dict:
                cva.drawArea(cimg, measurement_dict["jet_area"]["areas"], measurement_dict["jet_area"]["contour_indices"], shape["left_contour"], shape["right_contour"],text)
            if "jv" in annotate and "jet_volume" in measurement_init_dict:
                cva.drawVolume(cimg,  measurement_dict["jet_volume"]["volumes"], measurement_dict["jet_volume"]["contour_indices"], shape["left_contour"], shape["right_contour"],text)
            if "l" in annotate and y_min_l != 0 and y_min_r != 0:
                if y_min_r == y_min_l:
                    cva.lineBetweenPoints(cimg,([0,cimg.shape[1]],[y_min_l,y_min_r])) 
                else:
                    cva.lineBetweenPoints(cimg,([0,cimg.shape[1]],[y_min_l,y_min_r]),color=(255,0,255)) 

        return cimg
    
    def listLabels(label_path):
        label_dict = {}
        labels = []
        with open(label_path, "r") as csvfile:
            reader = csv.reader(csvfile)
            i = 0 
            for row in reader:
                #print(row)
                if i==0: 
                    i += 1
                    if row[0] != "Image":
                        #print("Error - Unknown table header")
                        break
                    for item in row[1:]:
                        labels.append(item)
                    #print("labels", labels)

                else:
                    img_nr = int(row[0])
                #print(img_nr)
                    label_dict[img_nr] = {}
                    j = 0
                    for item in row[1:]:
                        #print(labels[j])
                        label_dict[img_nr][labels[j]] = item
                        j += 1
                    #print(lable_dict[img_nr])
                # print(lable_dict[1]['Speed'])
        return label_dict, labels

    def checkLabelAnnotateList(annotate_labels, labels):
        new_annotate_labels = []
        for item in annotate_labels:
            if item in labels or item == "Image":
                new_annotate_labels.append(item)
            else:
                print(item, "not part of available labels - removed it from annotation list")
        return new_annotate_labels

    def annotateImgLabels(cimg, annotate_labels, label_dict, img_nr):
        x = 50
        y = 50
        color_text = (170,119,68)
        for item in annotate_labels:
            if img_nr in label_dict.keys():
                if item == "Image":
                    cv.putText(cimg, "Image No: " + str(img_nr), (x,y), cv.FONT_HERSHEY_SIMPLEX, 1, color_text, 2, cv.LINE_AA)
                # print("put text")
                else:
                    cv.putText(cimg, item + ": " + "%.2f" %(float(label_dict[img_nr][item])), (x,y), cv.FONT_HERSHEY_SIMPLEX, 1, color_text, 2, cv.LINE_AA)
                y += 40 
        return cimg

    def showImg(cimg, auto_close, scaler):
        cv.resizeWindow("Image", int(cimg.shape[1]*scaler), int(cimg.shape[0]*scaler))
        cv.imshow("Image",cimg) 
        if not auto_close:
            cv.waitKey(0)
        else:
            cv.waitKey(1) 
        return

    def initVideoWriter(out_path, frame_rate, size):
        output_file_name = "CV_" + "video.mp4"
        output_file_path = os.path.join(out_path, output_file_name)
        if os.path.exists(output_file_path) == False:
            vidout  = cv.VideoWriter(output_file_path, cv.VideoWriter_fourcc(*'mp4v'), frame_rate, size)
        else:
            print("Unable to create video file or file already exists")
            vidout = ""
        return vidout
    
    def saveFrame(img=[], img_no=0, out_path="", mode="save"):
        if mode == "init":
            image_out_path = fh.createSubDir(out_path,"AnnotatedImages")
            if not image_out_path:
                print("Unable to create sub-directory for images.")
            return image_out_path
        else:
            output_file_name = "CV_annotated_" + "image_"+str(img_no)+".jpg"
            #output_file_name = "CV_annotated_" + "image_"+str(img_no)+".bmp"
            output_file_path = os.path.join(out_path, output_file_name)
            cv.imwrite(output_file_path, img)
            return
        

    ### input arguments  ###
    mode = "n"
    # verbose mode
    verbose = False
    # save shapes
    save_shape = False
    # save measurements
    save_measurement = False
    #show image 
    show_img = False
    #auto close image 
    auto_close = 0
    # image annotations
    annotate = []
    # annotate with text
    text = False
    # annotate image labels
    annotate_labels = []
    #img size scaler 
    scaler = 1
    # save annotated images as video
    save_video = 0
    # save annotated images as images
    save_frame = False


    ### input range arguments ###
    # CV analysis, img number start, stop, e.g. to only analyse certain number of images 
    # Relative image numbers => sorted image names, count, not image number in image name 
    start_idx = 1 #we start counting at 1 in this case
    stop_idx = -1 #inclusive => e.g. 6 -> image number 6 will be analysed, -1 -> all images in folder
    nr_list = [] #list of specific image numbers to analyse

    #init data file path
    init_path = ""
    #measurement init data file path
    measurement_init_path = ""
    # shape path
    shape_path = ""
    # lable path
    label_path = ""

    ### output path arguments ###
    out_path = ""
    # pre define measurement and shape output csv file variables, for exitGracefully()
    measurement_out_csv = ""
    shape_out_csv = ""

    #pre-allocation of output csv headers 
    #shape csv header 
    shape_csv_header = ["Image", "shape_file_name"]
    #measurement csv header
    #measurement_csv_header = ["Image", "shape_file_name", "measurement_file_name"]
    measurement_csv_header = ["Image"]

    # Get arguments from command line 
    try:
        opts, args = getopt.getopt(argv, "hM:i:I:o:b:e:S:j:vsmtca:d:rFV:l:L:O:",[])
    except getopt.GetoptError:
        print('prog.py "-M"/"--mode" <mode>, "-i"/"--initfile" <initfile>,  "-I"/"--imagepath" <Imagefolder>, "-o"/"--outputpath" <outputfolder>, "-S"/"--shapepath" <shapefolder>, "-j"/"--measurementinitfile" <measurementinitfile>, "-b"/"--startnumber" <imagestartnumber>, "-e"/"--endnumber" <imageendnumber>, "-s"/"--saveshape", "-m"/"--savemeasurement", "-v"/"--verbose", "-t"/"--showimage", "-d"/"--imagesizescaler" <scaler>, "-c"/"--autoclose", "-a"/"--annotate" <annotations>, "-r"/"--annotateresult", "-V"/"--savevideo" <framerate>, "-F"/"--saveframe", "-L"/"--lablepath" <path>, "-l"/"--annotatelabels" <annotation>, "-O" <imagenumerlist> ')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('prog.py "-M"/"--mode" <mode>, "-i"/"--initfile" <initfile>,  "-I"/"--imagepath" <Imagefolder>, "-o"/"--outputpath" <outputfolder>, "-S"/"--shapepath" <shapefolder>, "-j"/"--measurementinitfile" <measurementinitfile>, "-b"/"--startnumber" <imagestartnumber>, "-e"/"--endnumber" <imageendnumber>, "-s"/"--saveshape", "-m"/"--savemeasurement", "-v"/"--verbose", "-t"/"--showimage", "-d"/"--imagesizescaler" <scaler>, "-c"/"--autoclose", "-a"/"--annotate" <annotations>, "-r"/"--annotateresult", "-V"/"--savevideo" <framerate>, "-F"/"--saveframe", "-L"/"--lablepath" <path>, "-l"/"--annotatelabels" <annotation>, "-O" <imagenumerlist> ')
            sys.exit()
        elif opt in ("-M", "--mode"):    
            mode = arg
        elif opt in ("-i", "--initfile"):
            init_path = arg
        elif opt in ("-I", "--imagepath"):
            img_path = arg
        elif opt in ("-o", "--outputpath"):
            out_path = arg
        elif opt in ("-b", "--startnumber"):
            start_idx = int(arg)
        elif opt in ("-e", "--endnumber"):
            stop_idx = int(arg)
        elif opt in ("-S", "--shapepath"):
            shape_path = arg 
        elif opt in ("-j", "--measurementinitfile"):
            measurement_init_path = arg 
        elif opt in ("-v", "--verbose"):
            verbose = True
        elif opt in ("-s", "--saveshape"):
            save_shape = True
        elif opt in ("-m", "--savemeasurement"):
            save_measurement = True
        elif opt in ("-t", "--showimage"):
            show_img = True
        elif opt in ("-c", "--autoclose"):
            auto_close = 0.1
        elif opt in ("-a", "--annotate"):
            annotate = arg.split(",")
        elif opt in ("-d", "--imagesizescaler"):
            scaler = float(arg)
        elif opt in ("-r", "--annotateresult"):
            text = True
        elif opt in ("-V", "--savevideo"):
            save_video = int(float(arg))
        elif opt in ("-F", "--saveframe"):
            save_frame = True
        elif opt in ("-l", "--annotatelabels"):
            annotate_labels = arg.split(",")
        elif opt in ("-L", "--lablepath"):
            label_path = arg
        elif opt in ("-O"):
            nr_list = arg
            nr_list = arg.split(',')


    if show_img:
        cv.namedWindow("Image", cv.WINDOW_GUI_EXPANDED)

    if mode == "n":
        print("No valid mode selected")
        print("Valid modes: i - Initialisation, s - Jet shape extraction, m - Measurement from shape data, sm - Jet shape extraction AND measurement")
        exitGracefully()
    #If we try to go into initialisation mode..
    elif mode == "i":
        #check if we got everything that we need 
        if out_path == "" or img_path == "":
            print("Not all required arguments given.")
            print("We need an output path where the init file will be saved and a image path for the image we want to use in the initialisation process")
            exitGracefully()
        else:
            cvinit.main(img_path, out_path)
    #If we try to extract the shapes
    elif mode == "s":
        #check if we got everything that we need 
        if not init_path or not img_path:
            print("Not all required arguments given.")
            print("We need an init file path and an image folder path")
            exitGracefully()
        if (save_shape and not out_path):
            print("An output folder path must be specified to save shapes.")
            exitGracefully()
        else:
            if save_shape and out_path: 
                print("Saving shapes.")
            else:
                print("Not saving shapes. Provide -o <output_path> and set save flag -s , to save shapes")
        if save_frame:
            if not annotate:
                print("Not saving images. Provide -a <annotations> to create annotated image.")
                save_frame = False
            elif not out_path:
                print("Not saving images.  Provide -o <output_path> to save annotated images.")
                save_frame = False
            else:
                print("Saving annotated images.")
        if save_video != 0:
            if not annotate:
                print("Not saving video. Provide -a <annotations> to create annotated image.")
                save_video = 0
            elif not out_path:
                print("Not saving video.  Provide -o <output_path> to save video.")
                save_video = 0
            else:
                print("Saving video.")
        



        # load init data
        init_dict = loadInit(init_path)
        #load images 
        img_dict, img_keys, img_count = listImgs(img_path)
        #if we want to save, create a sub-dir and csv file for our output 
        if save_shape:
            shape_out_path, shape_out_csv, shape_out_csv_writer = createShapeOut(out_path) 
            shape_out_csv_writer.writerow(shape_csv_header) #write the header into our csv file 
        if save_frame:
            image_out_path = saveFrame(out_path=out_path, mode="init") #creates output directory for images
            if not image_out_path:
                save_frame = False
        #if we want to annotate labels, load them and check requested annotations are valid
        if label_path and annotate_labels:
            label_dict, labels = listLabels(label_path)
            annotate_labels = checkLabelAnnotateList(annotate_labels, labels)
        elif annotate_labels == ["Image"]: 
            annotate_labels = ["Image"]
            label_dict = {}



        #setup iterator 
        key_iterator, iterator_steps  = getIterator(img_count)

        print("Start extracting shapes...")
        mtic = time.time() #to keep track of time 
        invalid_count = 0
        videoinit = False
        while True:
            #get next image key
            img_key = img_keys[key_iterator]
            #load the image
            img, image_loaded = loadImg(img_key)
            if image_loaded: #only if succsesful, continue with shape extraction
                thres_img, shape, shape_status, all_contours, y_min_l, y_min_r = extractShape(img, init_dict) # try to extract jet contour and midline 
                if not shape_status:
                    invalid_count += 1
                if save_shape: # if we want to save the output
                    _ = saveShapeNEW(shape, img_key, shape_out_path, shape_out_csv_writer, verbose)
                if verbose:
                    print("Analysed ", key_iterator - start_idx + 2, "of ", iterator_steps, " images")
                
                if annotate or annotate_labels:
                    cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR) #create color img 
                   
                if annotate: #creates annotated image
                   # cimg_width = cimg.shape[1] ### TO GET JET SHAPE ON WHITE BACKGROUDN IMAGE
                   # cimg_height = cimg.shape[0]
                   # timg = np.zeros([cimg_height,cimg_width,3],dtype=np.uint8)
                   # timg[:] = 255
                    cimg = annotateImg(cimg, annotate, shape_status, all_contours, shape, {}, init_dict, text, y_min_l, y_min_r)
                if annotate_labels:
                    cimg = annotateImgLabels(cimg, annotate_labels, label_dict, img_key)
                if show_img: #shows image
                    if annotate or annotate_labels:
                        showImg(cimg, auto_close, scaler)
                    else:
                        showImg(img, auto_close, scaler)
                if save_frame: #saves annotated image
                    saveFrame(cimg, img_key, image_out_path)
                if save_video != 0:
                    if not videoinit:
                        vidout = initVideoWriter(out_path, save_video, (cimg.shape[1],cimg.shape[0]))
                        videoinit = True
                        if vidout == "":
                            save_video = 0
                        else: 
                            vidout.write(cimg)
                    else:
                        vidout.write(cimg)


            key_iterator += 1 #count up iterator
            if stop_idx == -1: 
                if key_iterator > img_count-1:
                    break #stop if we reached end 
            else:
                if (key_iterator > stop_idx-1) or (key_iterator > img_count-1):
                    break #stop if we reached end 

        if save_video !=0:
            vidout.release()
        mtoc = time.time()
        print("Finished shape extraction in:", mtoc-mtic, "seconds aka.", (mtoc-mtic)/60, "minutes.")
        print("On average", (mtoc-mtic)/iterator_steps, "seconds per image.")
        print("Number of failed analyses:", invalid_count)
        
    
    #If we try to measure based on knows shapes
    elif mode == "m": 
        #check if we got everything that we need 
        if not init_path or not measurement_init_path or not shape_path:
            print("Not all required arguments given.")
            print("We need an init file path, an measurement init file path and a shape folder path")
            exitGracefully()
        if (save_measurement and not out_path):
            print("An output folder path must be specified to save measurements.")
            exitGracefully()
        else:
            if save_measurement and out_path: 
                print("Saving measurements.")
            else:
                print("Not saving measurements. Provide -o <output_path> and set save flag -m , to save measurements")
        if show_img and not img_path:
            print("Can't show images, image path is missing, Provide -I <image_path>.")
            show_img = False

        if save_frame:
            if not img_path:
                print("Can't save annotated images, image path is missing, Provide -I <image_path>.")
                save_frame = False
            else:
                if not annotate:
                    print("Not saving images. Provide -a <annotations> to create annotated image.")
                    save_frame = False
                elif not out_path:
                    print("Not saving images.  Provide -o <output_path> to save annotated images.")
                    save_frame = False
                else:
                    print("Saving annotated images.")
        if save_video != 0 and not img_path:
            print("Can't save video, image path is missing, Provide -I <image_path>.")
            save_video = 0
        else:
            if not annotate:
                print("Not saving video. Provide -a <annotations> to create annotated image.")
                save_video = 0
            elif not out_path:
                print("Not saving video.  Provide -o <output_path> to save video.")
                save_video = 0
            else:
                print("Saving video.")

        # load init data
        init_dict = loadInit(init_path)

        # load measurement init data
        measurement_init_dict = loadMeasurementInit(measurement_init_path)      
        # get flags for measurement references 
        measurement_flags = getMeasurementReferenceFlags(measurement_init_dict)     
        #list shapes
        shape_dict, shape_keys, shape_count = listShapes(shape_path)
        #if we want to annotate labels, load them and check requested annotations are valid
        if label_path and annotate_labels:
            label_dict, labels = listLabels(label_path)
            annotate_labels = checkLabelAnnotateList(annotate_labels, labels)
        elif annotate_labels == ["Image"]: 
            annotate_labels = ["Image"]
            label_dict = {}
        #prepare save output
        if save_measurement:
            measurement_csv_header = setupMeasurementsHeader(measurement_init_dict) #create CSV file header
            measurement_out_csv, measurement_out_csv_writer = createMeasurementOut(out_path) #create output csv
            measurement_out_csv_writer.writerow(measurement_csv_header) #write the header into our csv file 
        if save_frame:
            image_out_path = saveFrame(out_path=out_path, mode="init") #creates output directory for images
            if not image_out_path:
                save_frame = False



        #setup iterator 
        key_iterator, iterator_steps  = getIterator(shape_count)

        if show_img or save_frame or save_video!= 0:
            img_dict, img_keys, img_count = listImgs(img_path) #load images 
            if not img_dict:
                print("Unable to list files from image folder path")
                show_img = False


        print("Start taking measurements...")
        mtic = time.time() #to keep track of time 
        invalid_count = 0
        videoinit = False
        listiterator = False
        if len(nr_list) != 0:
            listiterator = True  
        listiterator_i = 0
        while True:
            #get key for next image from dict 
            if listiterator == False:
                shape_key = shape_keys[key_iterator]
            else:
                shape_key = shape_keys.index(int(nr_list[listiterator_i]))
                


            if verbose:
                print("shape_key(img_nr)", shape_key)
            
            #load shape
            shape = loadShape(shape_key)
            
            #take measurements and save them 
            measurement_dict = takeMeasurements(measurement_flags, shape, shape_key, shape_dict[shape_key][0], init_dict, measurement_init_dict, verbose)
            if save_measurement:
                saveMeasurements(measurement_dict, measurement_out_csv_writer, verbose)

            image_loaded = False
            if show_img or save_frame or save_video!= 0: #load image
                img, image_loaded = loadImg(shape_key)

            if image_loaded:
                if annotate or annotate_labels:
                    cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR) #create color img 
                if annotate: #creates annotated image
                    cimg = annotateImg(cimg, annotate, True, {}, shape, measurement_dict, init_dict, text, y_min_l=init_dict["collector_left"], y_min_r=init_dict["collector_right"])
                if annotate_labels:
                    cimg = annotateImgLabels(cimg, annotate_labels, label_dict, shape_key)
                if show_img: #shows annotated image
                    if annotate:
                        showImg(cimg, auto_close, scaler)
                    else:
                        showImg(img, auto_close, scaler)
                if save_frame: #saves annotated image
                    saveFrame(cimg, shape_key, image_out_path)
                if save_video != 0:
                    if not videoinit:
                        vidout = initVideoWriter(out_path, save_video, (cimg.shape[1],cimg.shape[0]))
                        videoinit = True
                        if vidout == "":
                            save_video = 0
                        else: 
                            vidout.write(cimg)
                    else:
                        vidout.write(cimg)
            else:
                print("Failed to load image.")

            if listiterator == False:
                key_iterator += 1 #count up iterator
                if stop_idx == -1: 
                    if key_iterator > shape_count-1:
                        break #stop if we reached end 
                else:
                    if (key_iterator > stop_idx-1) or (key_iterator > shape_count-1):
                        break #stop if we reached end 
            else:
                listiterator_i += 1
                if listiterator_i > len(nr_list)-1:
                    break
        
        if save_video !=0:
            vidout.release()
        mtoc = time.time()
        print("Finished measurements in:", mtoc-mtic, "seconds aka.", (mtoc-mtic)/60, "minutes.")
        print("On average", (mtoc-mtic)/iterator_steps, "seconds per shape.")

     #If we try to extract the shapes and at the same time take measurements 
    elif mode == "sm":
        #check if we got everything that we need 
        if not init_path or not img_path or not measurement_init_path: 
            print("Not all required arguments given.")
            print("We need an init file path, an image folder path and a measurement init file math")
            exitGracefully()
        if (save_shape and not out_path) or (save_measurement and not out_path):
            print("An output folder path must be specified to save shapes and/or measurements.")
            exitGracefully()
        else: 
            if save_shape and out_path: 
                print("Saving shapes.")
            else:
                print("Not saving shapes. Provide -o output_path and set save flag -s , to save shapes")
            if save_measurement and out_path: 
                print("Saving measurements.")            
            else:
                print("Not saving measurements. Provide -o output_path and set save flag -m , to save measurements")
        
        if save_frame:
            if not annotate:
                print("Not saving images. Provide -a <annotations> to create annotated image.")
                save_frame = False
            elif not out_path:
                print("Not saving images.  Provide -o <output_path> to save annotated images.")
                save_frame = False
            else:
                print("Saving annotated images.")
        if save_video != 0:
            if not annotate:
                print("Not saving video. Provide -a <annotations> to create annotated image.")
                save_video = 0
            elif not out_path:
                print("Not saving video.  Provide -o <output_path> to save video.")
                save_video = 0
            else:
                print("Saving video.")


    
        # load init data
        init_dict = loadInit(init_path)
        #load images 
        img_dict, img_keys, img_count = listImgs(img_path)
        # load measurement init data
        measurement_init_dict = loadMeasurementInit(measurement_init_path) 
        #print(measurement_init_dict) 
        # get flags for measurement references 
        measurement_flags = getMeasurementReferenceFlags(measurement_init_dict)
        #print(measurement_flags)
        #if we want to annotate labels, load them and check requested annotations are valid
        if label_path and annotate_labels:
            label_dict, labels = listLabels(label_path)
            annotate_labels = checkLabelAnnotateList(annotate_labels, labels)
        elif annotate_labels == ["Image"]: 
            annotate_labels = ["Image"]
            label_dict = {}
        #if we want to save, create a sub-dir and csv file for our output 
        if save_shape:
            shape_out_path, shape_out_csv, shape_out_csv_writer = createShapeOut(out_path) 
            shape_out_csv_writer.writerow(shape_csv_header) #write the header into our csv file 
        if save_measurement:
            measurement_csv_header = setupMeasurementsHeader(measurement_init_dict) #create CSV file header
            measurement_out_csv, measurement_out_csv_writer = createMeasurementOut(out_path) #create output csv/sub-dir
            measurement_out_csv_writer.writerow(measurement_csv_header) #write the header into our csv file 
        if save_frame:
            image_out_path = saveFrame(out_path=out_path, mode="init") #creates output directory for images
            if not image_out_path:
                save_frame = False
        #setup iterator 
        key_iterator, iterator_steps  = getIterator(img_count)

        print("Start extracting shapes and taking measurements...")
        mtic = time.time() #to keep track of time 
        invalid_count = 0
        videoinit = False
        listiterator = False
        if len(nr_list) != 0:
            listiterator = True  
        listiterator_i = 0
        while True:
            #get next image key
            if listiterator == False:
                img_key = img_keys[key_iterator]
            else:
                img_key = int(nr_list[listiterator_i])
            #load the image
            img, image_loaded = loadImg(img_key)
            if image_loaded: #only if succsesful, continue with shape extraction
                thres_img, shape, shape_status, all_contours, y_min_l, y_min_r = extractShape(img, init_dict) # try to extract jet contour and midline 
                shape_file_name = "Not saved." 

                

                # NEEDS FIXING
                if save_shape: # if we want to save the output
                    _ = saveShapeNEW(shape, img_key, shape_out_path, shape_out_csv_writer, verbose)
                    
                
                measurement_dict = {}  
                if not shape_status:
                    invalid_count += 1
                else:                 
                    measurement_dict = takeMeasurements(measurement_flags, shape, img_key, shape_file_name, init_dict, measurement_init_dict, verbose)
                    if save_measurement:
                        saveMeasurements(measurement_dict, measurement_out_csv_writer, verbose)
            
                if verbose:
                    print("Analysed ", key_iterator - start_idx + 2, "of ", iterator_steps, " images")

                if annotate or annotate_labels:
                    cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR) #create color img 
                   
                if annotate: #creates annotated image
                    cimg = annotateImg(cimg, annotate, shape_status, all_contours, shape, measurement_dict, init_dict, text, y_min_l, y_min_r)
                if annotate_labels:
                    cimg = annotateImgLabels(cimg, annotate_labels, label_dict, img_key)
                if show_img: #shows annotated image
                    if annotate or annotate_labels:
                        showImg(cimg, auto_close, scaler)
                    else:
                        showImg(img, auto_close, scaler)
                if save_frame: #saves annotated image
                    saveFrame(cimg, img_key, image_out_path)
                if save_video != 0:
                    if not videoinit:
                        vidout = initVideoWriter(out_path, save_video, (cimg.shape[1],cimg.shape[0]))
                        videoinit = True
                        if vidout == "":
                            save_video = 0
                        else: 
                            vidout.write(cimg)
                    else:
                        vidout.write(cimg)

            if listiterator == False:
                key_iterator += 1 #count up iterator
                if stop_idx == -1: 
                    if key_iterator > img_count-1:
                        break #stop if we reached end 
                else:
                    if (key_iterator > stop_idx-1) or (key_iterator > img_count-1):
                        break #stop if we reached end 
            else:
                listiterator_i += 1
                if listiterator_i > len(nr_list)-1:
                    break

        if save_video !=0:
            vidout.release()
        mtoc = time.time()
        print("Finished shape extraction and measurements in:", mtoc-mtic, "seconds aka.", (mtoc-mtic)/60, "minutes.")
        print("On average", (mtoc-mtic)/iterator_steps, "seconds per image.")
        print("Number of failed analyses:", invalid_count)


    exitGracefully()



if __name__ == '__main__':
    main(sys.argv[1:])