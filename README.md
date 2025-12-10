# MEW_CV_Publication

This code provides implementation of the computer vision (CV) based melt electrowriting (MEW) jet measurement and 3D scaffold reconstruction techniques proposed in "PAPER TITLE". 
The publication can be found at: 

PAPER INFORMATION

The code is free to use for research and non-commercial purposes. Please cite if you do so.

## Overview
### Computer vision code
The CV code loads images and extracts the jet shape and selected measurements through a number of processing steps. It is expected that the images are distrortion free and that the pixel scale is uniform over the entire image. This may require image pre-processing, depending on the imaging setup. The processing pipeline follows two major steps. First, the jet shape which is defined by the left edge, the right edge and the midline of the jet is extracted. Second, based on the jet shape, measurements of the jet diameter, jet angle, jet length, jet area and jet vector can be taken. The two steps can be processed individually or combined. The code provides a command line interface to specify input- and output-files as well as a number of options. Measurements settings are specified in a excel sheet.

### 3D scaffold reconstruction
The 3D scaffold reconstruction code loads specific measurements, that were taken with the CV code, and uses them to reconstruct and 3D plot the printed scaffold structure. This is implemented in a Matlab live-script for ease of use. 

## Setup
### Computer vision code
1) Clone the "Python" folder.
2) Ensure all required packages are installed, see list below. Refer to the individual package installation instructions.
3) Ensure all images that are to be processed are in a folder with no other files. The image naming must follow: "identtext1_identtext2_image_IMAGENO_id_IMAGEID.jpg". identtext1 and identtext2 can be arbitrary but must be set. For example "jet_camera". IMAGENO is the image number. If there are duplicates, only the image loaded last will be processed. IMAGEID must be set and should be a unique identification number. The IMAGEID does not influence the CV calculations but is required for later processing of the results in the here provided Matlab code.
5) If outputs/results are to be saved, create an ouput folder. 
6) Run the "main.py" code, see "Use" section below for details.

### 3D scaffold reconstruction


## Use
### Computer vision code
The CV code can be manipulated via command line arguments, see list below.
Typical workflow example:
1) Creation of a initialisation file.
Run: py main.py -M i -I <imagepath>

2) Define measurements.
Open the "Excel_Measurement_Spec.xlsx" file. Specify the measurements and export as .csv file as described in the excel workbook.

3) Extract jet shapes and take measurements.
Prepare image folder and output folder, as described in "Setup" section above.
Run: py main.py -M sm -I <imagefolderpath> -i <initfilepath> .... => make nice images when paper prep and then put in here as example.


| Command line arguments | Note |
| --- | --- | 
| -M, --mode \<mode>| Mode of operation. i = Initialisation. s = Shape extraction from image file(s). m = Measurement from shape file(s). sm = Shape extraction from image file(s) and measurement from extracted shape.|
| -i, --initfile \<path> | File path to a init-file that was created in initialisation mode.|
| -I, --imagepath \<path> | Path to the image(s) directory.|
| -o, --outputpath \<path> | Path to the directory in which shapes, measurements or init-files are saved.|
| -S, --shapepath \<path> | Path to the shape(s) directory.|
| -j, --measurementinitfile \<path> | File path to a measurement-init file that contains the measurement specifications.|
| -b, --startnumber \<number> | Start number of file to analyse, default = 1|
| -e, --endnumber \<number> | End number (inclusive) of file to analyse, default = last available file. |
| -s, --saveshape | If set, extracted shapes are saved to outputpath. |
| -m, --savemeasurement | If set, measurements are saved to outputpath.|
| -v, --verbose | If set, detailed status information and measurements are outputted to command line. | 
| -t, --showimage | If set, processed image is shown. |
| -d, --imagesizescaler \<scale>| Scales the shown image by the provided number., e.g. 0.5 -> half size. |
| -c, --autoclose | If set, image window is updated with next image as soon as processing of past image finished. |
| -a, --annotate \<annotation> | Annotates the shown image with: e - jet edges, m - jet midline, c - all found contours, ja - measured jet angle(s), jd - measured jet diameter(s), jL - measured jet length(s), jA - measured jet area(s), jv - measured jet vector(s). If more than one is required, provide as comma separated text, e.g.: e,m,jd |
| -r, --annotateresult| Adds the measurement as text to the image annotations. |
| -V, --savevideo \<framerate> | Creates a video made up of the processed images with annotations. Framerate (framses/second) is defined by the provided number. |
| -F, --saveframe | If set, processed image with annotations is saved.
| -O | nr1,nr2,... List of image numbers to analyse, allows only specific images from the image dictionary to be analysed. |

### 3D scaffold reconstruction



## Dependencies 
A number of software packages are used by the code as listed below. 
The list includes the versions that were used. Newer versions are likely compatible too.
### Computer vision code

| Software / Package | Version |
| --- | --- |
| Python | 3.11.2 |
| opencv-contrib-python| 4.9.0.80 |
| scikit-image | 0.22.0 |
| NumPy | 1.26.4 |
| SciPy | 1.12.0 |
| sknw | 0.15 |
| networkx | 3.2.1 |
