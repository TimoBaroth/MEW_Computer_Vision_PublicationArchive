import pickle
import os
import csv

def listImagesNEW(img_path):
    img_dict = {}
    dict_keys = []
    img_count = 0
    try:
        file_list = os.listdir(img_path) #get list of files
        #We expect the files to be named like this: "jet_cam_image_NR_id_ID.jpg" with NR the number of the frame and ID the ID 
        for item in file_list: # go through list 
            splitres = item.split('.')
            img_nr = int(splitres[0]) #get image number of item  
            img_dict[img_nr] = [item] #save image number and corresponding file path and id in dict 
        img_dict = dict(sorted(img_dict.items())) #sort dict by image number 
        dict_keys = list(img_dict.keys()) #create list with the dict keys
        img_count = len(dict_keys) #number of images in our folder is equal to number of keys in our dict
    except:
        pass
    return img_dict, dict_keys, img_count

def listImages(img_path):
    img_dict = {}
    dict_keys = []
    img_count = 0
    try:
        file_list = os.listdir(img_path) #get list of files
        #We expect the files to be named like this: "jet_cam_image_NR_id_ID.jpg" with NR the number of the frame and ID the ID 
        for item in file_list: # go through list 
            splitres = item.split('_')
            img_nr = int(splitres[3]) #get image number of item  
            img_id = splitres[5].split('.')[0] #get image ID of item
            img_dict[img_nr] = [item, img_id] #save image number and corresponding file path and id in dict 
        img_dict = dict(sorted(img_dict.items())) #sort dict by image number 
        dict_keys = list(img_dict.keys()) #create list with the dict keys
        img_count = len(dict_keys) #number of images in our folder is equal to number of keys in our dict
    except:
        pass
    return img_dict, dict_keys, img_count

def listImagesByID(img_path):
    img_dict = {}
    try:
        file_list = os.listdir(img_path) #get list of files
        #We expect the files to be named like this: "jet_cam_image_NR_id_ID.jpg" with NR the number of the frame and ID the ID 
        for item in file_list: # go through list 
            splitres = item.split('_')
            img_id = int(splitres[5].split('.')[0]) #get image ID of item
            img_dict[img_id] = [item] #save image number and corresponding file path and id in dict 
    except:
        pass
    return img_dict

def listShapesNEW(shape_path):
    shape_dict = {}
    dict_keys = []
    shape_count = 0
    try:
        file_list = os.listdir(shape_path) #get list of files
        #We expect the files to be named like this: "CV_shape_image_NR_id_ID.xz" with NR the number of the frame and ID the ID 
        #Following: "jet_cam_image_NR_id_ID.jpg" with NR the number of the frame and ID the ID 
        for item in file_list: # go through list 
            splitres = item.split('_')
            if splitres[-1].split('.')[-1] != "xz":
                print("Error - Found non lzma(.xz) files in shape folder")
                return shape_dict, dict_keys, shape_count
            img_nr = int(splitres[3].split('.')[0]) #get image number of item  
            shape_dict[img_nr] = [item] #save image number and corresponding file path and id in dict 
        shape_dict = dict(sorted(shape_dict.items())) #sort dict by image number 
        dict_keys = list(shape_dict.keys()) #create list with the dict keys
        shape_count = len(dict_keys) #number of images in our folder is equal to number of keys in our dict
    except:
        pass 
    
    return shape_dict, dict_keys, shape_count




def listShapes(shape_path):
    shape_dict = {}
    dict_keys = []
    shape_count = 0
    try:
        file_list = os.listdir(shape_path) #get list of files
        #We expect the files to be named like this: "CV_shape_image_NR_id_ID.xz" with NR the number of the frame and ID the ID 
        #Following: "jet_cam_image_NR_id_ID.jpg" with NR the number of the frame and ID the ID 
        for item in file_list: # go through list 
            splitres = item.split('_')
            if splitres[-1].split('.')[-1] != "xz":
                print("Error - Found non lzma(.xz) files in shape folder")
                return shape_dict, dict_keys, shape_count
            img_nr = int(splitres[3]) #get image number of item  
            img_id = splitres[5].split('.')[0] #get image ID of item
            shape_dict[img_nr] = [item, img_id] #save image number and corresponding file path and id in dict 
        shape_dict = dict(sorted(shape_dict.items())) #sort dict by image number 
        dict_keys = list(shape_dict.keys()) #create list with the dict keys
        shape_count = len(dict_keys) #number of images in our folder is equal to number of keys in our dict
    except:
        pass 
    return shape_dict, dict_keys, shape_count


def loadInitData(init_path):
    init_data = {}
    try:
        with open(init_path, 'rb') as fp:
                init_data = pickle.load(fp)
    except: 
        print("Unable to load initialisation file")
    return init_data

def createSubDir(path, name):
    try:
        sub_dir = os.path.join(path, name)
        os.mkdir(sub_dir)
    except:
        sub_dir = ""
        pass
    return sub_dir

def createCSV(path, name):
    csv_file = os.path.join(path, name + '.csv')
    csvf = ""
    csvwriter = ""
    if os.path.exists(csv_file) == False:
        csvf = open(csv_file, "w", newline='') 
        csvwriter = csv.writer(csvf, dialect='excel')
    return csvf, csvwriter 

def loadMeasurmentInitCSV(path):
    measurement_init_dict = {}
    measurement_init_dict["general"] = {}
    try:
        with open(path, "r") as csvfile:
            reader = csv.reader(csvfile)
            i = 0 
            for row in reader:
                if i==0 and row != ['Measurement', 'Reference', 'MeasurementPositions']:
                    print("Error - Unknown table header, expecting: Measurement, Reference, MeasurementPositions")
                    break
                elif row[0] == "JetLag" and row[2]:
                    measurement_init_dict["jet_lag"] = {}
                    measurement_init_dict["jet_lag"]["mode"] = row[1][0:3]
                    measurement_init_dict["jet_lag"]["positions"] = [int(item.split(".")[0]) for item in row[2].split(";")]
                elif row[0] == "JetAngle" and row[2]:
                    measurement_init_dict["jet_angle"] = {}
                    measurement_init_dict["jet_angle"]["mode"] = row[1][0:3]
                    measurement_init_dict["jet_angle"]["positions"] = [int(item.split(".")[0]) for item in row[2].split(";")]
                elif row[0] == "JetDiameter" and row[2]:
                    measurement_init_dict["jet_diameter"] = {}
                    measurement_init_dict["jet_diameter"]["mode"] = row[1][0:3]
                    measurement_init_dict["jet_diameter"]["positions"] = [int(item.split(".")[0]) for item in row[2].split(";")]
                elif row[0] == "JetLength" and row[2]:
                    measurement_init_dict["jet_length"] = {}
                    measurement_init_dict["jet_length"]["mode"] = row[1][0:3]
                    measurement_init_dict["jet_length"]["positions"] = [int(item.split(".")[0]) for item in row[2].split(";")]
                elif row[0] == "JetArea" and row[2]:
                    measurement_init_dict["jet_area"] = {}
                    measurement_init_dict["jet_area"]["mode"] = row[1][0:3]
                    measurement_init_dict["jet_area"]["positions"] = [int(item.split(".")[0]) for item in row[2].split(";")]
                elif row[0] == "JetVolume" and row[2]:
                    measurement_init_dict["jet_volume"] = {}
                    measurement_init_dict["jet_volume"]["mode"] = row[1][0:3]
                    measurement_init_dict["jet_volume"]["positions"] = [int(item.split(".")[0]) for item in row[2].split(";")]

                if row[0] == "PixelSize":
                    measurement_init_dict["general"]["pixel_size"] = float(row[1])
                elif row[0] == "DiameterNormalApproxDist":
                    measurement_init_dict["general"]["diameter_approx_distance"] = int(row[1])

                i += 1
    except:
        print("Unable to load measurement initialisation file")
    return measurement_init_dict

