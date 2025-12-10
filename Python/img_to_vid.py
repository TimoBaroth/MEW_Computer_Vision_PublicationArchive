import cv2 as cv
import os
import csv
import sys, getopt
import datetime

def main(argv):

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
                    if row[0] != 'Image':
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


    def checkLabelAnnotateList(annotate_labels, lables):
        new_annotate_lables = []
        for item in annotate_labels:
            if item in lables or item == "Image":
                new_annotate_lables.append(item)
            else:
                print(item, "not part of available labels - removed it from annotation list")
        return new_annotate_lables



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
                    cv.putText(cimg, item + ": " + label_dict[img_nr][item], (x,y), cv.FONT_HERSHEY_SIMPLEX, 1, color_text, 2, cv.LINE_AA)
                y += 40 
        return cimg



    def listImagesBNEW(img_path):
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



    def listImagesB(img_path):
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

    def listImgs(img_path):
        img_dict, img_keys, img_count = listImagesBNEW(img_path)
        if not img_dict:
            print("Unable to list files from image folder path")
            exit() #if we failed to load it exit
        else:
            print("Image folder files listed. Total number of files:", img_count)
        return img_dict, img_keys, img_count 

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
           # print(img_dict[img_key])
            imgpath = os.path.join(img_path, img_dict[img_key][0])
            #imgpath = img_folder_path + "\\" + img_dict[img_key][0]
            img = cv.imread(imgpath,cv.IMREAD_GRAYSCALE) #load it 
            image_loaded = True
        return img, image_loaded

    def initVideoWriter(out_path, frame_rate, size):
        output_file_name = "CV_" + "video.mp4"
        output_file_path = os.path.join(out_path, output_file_name)
        if os.path.exists(output_file_path) == False:
            vidout  = cv.VideoWriter(output_file_path, cv.VideoWriter_fourcc(*'mp4v'), frame_rate, size)
        else:
            print("Unable to create video file or file already exists")
            vidout = ""
        return vidout



    img_path = ""
    out_path = ""
    label_path = ""

    annotate_labels = []

    start_idx = 1
    stop_idx = -1

    fps = -1


    try:
        opts, args = getopt.getopt(argv, "I:l:L:o:V:b:e:",[])
    except getopt.GetoptError:
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-I", "--imagepath"):
            img_path = arg
        elif opt in ("-l", "--annotatelabels"):
            annotate_labels = arg.split(",")
        elif opt in ("-L", "--lablepath"):
            label_path = arg
        elif opt in ("-o", "--outputpath"):
            out_path = arg
        elif opt in ("-V", "--savevideo"):
            fps = int(float(arg))
        elif opt in ("-b", "--startnumber"):
            start_idx = int(arg)
        elif opt in ("-e", "--endnumber"):
            stop_idx = int(arg)

    if fps == -1:
        print("FPS must be specified, -V <fps>")
        exit()
    
    if label_path and annotate_labels:
        label_dict, lables = listLabels(label_path)
        annotate_labels = checkLabelAnnotateList(annotate_labels, lables)
    elif annotate_labels == ["Image"]: 
        annotate_labels = ["Image"]
        label_dict = {}


    img_dict, img_keys, img_count = listImgs(img_path)
    key_iterator, iterator_steps  = getIterator(img_count)


    i = 0
    j = 0
    p = 0
    j_max = int(img_count/100)
    print("1% =", j_max, "images")
    print("Start creating video")
    while True:
        #get next image key
        img_key = img_keys[key_iterator]
        
        #load the image
        img, image_loaded = loadImg(img_key)
        
        if image_loaded: #only if succsesful, continue with shape extraction
            cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            if annotate_labels:
                cimg = annotateImgLabels(cimg, annotate_labels, label_dict, img_key)

            if i == 0:
                vidout = initVideoWriter(out_path, fps, (cimg.shape[1],cimg.shape[0]))
                if vidout == "":
                    exit()
                i = 1
            vidout.write(cimg)

        if j == j_max:
            p += 1
            j = 0
            print("Progress:", p, "%")

        j += 1

        key_iterator += 1 #count up iterator
        if stop_idx == -1: 
            if key_iterator > img_count-1:
                break #stop if we reached end 
        else:
            if (key_iterator > stop_idx-1) or (key_iterator > img_count-1):
                break #stop if we reached end 

    print("Done")


if __name__ == '__main__':
    main(sys.argv[1:])