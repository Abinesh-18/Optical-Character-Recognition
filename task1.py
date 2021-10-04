"""
Character Detection

The goal of this task is to implement an optical character recognition system consisting of Enrollment, Detection and Recognition sub tasks

Please complete all the functions that are labelled with '# TODO'. When implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them.

Do NOT modify the code provided.
Please follow the guidelines mentioned in the project1.pdf
Do NOT import any library (function, module, etc.).
"""


import argparse
import json
import os
import glob
import cv2
import numpy as np


def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if show:
        show_image(img)

    return img

def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--test_img", type=str, default="./data/test_img.png",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--character_folder_path", type=str, default="./data/characters",
        help="path to the characters folder")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args

def ocr(test_img, characters):
    print(characters)
    """Step 1 : Enroll a set of characters. Also, you may store features in an intermediate file.
       Step 2 : Use connected component labeling to detect various characters in an test_img.
       Step 3 : Taking each of the character detected from previous step,
         and your features for each of the enrolled characters, you are required to a recognition or matching.

    Args:
        test_img : image that contains character to be detected.
        characters_list: list of characters along with name for each character.

    Returns:
    a nested list, where each element is a dictionary with {"bbox" : (x(int), y (int), w (int), h (int)), "name" : (string)},
        x: row that the character appears (starts from 0).
        y: column that the character appears (starts from 0).
        w: width of the detected character.
        h: height of the detected character.
        name: name of character provided or "UNKNOWN".
        Note : the order of detected characters should follow english text reading pattern, i.e.,
            list should start from top left, then move from left to right. After finishing the first line, go to the next line and continue.
        
    """
    # TODO Add your code here. Do not modify the return and input arguments
    #print(characters)

    #print(test_img)
    TList=list()
    LList=list()
    TList = enrollment(characters)
    print(TList[1].name)
    charname = TList[2].name
    print("Enrollment complete")
    LList = detection(test_img)    
    print("Detection complete")
    
    resultsf = recognition(LList, TList)
    
    return resultsf
    
    raise NotImplementedError

def enrollment(characters):
    """ Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """

    # TODO: Step 1 : Your Enrollment code should go here
    class testdata:
      def __init__(self,name,charimage,shape):
         self.name= name
         self.charimage=charimage
         self.shape = shape
    len(characters)   
    testlist = []
    for charac, charac_img in characters:
      #cv2.imshow('img1',charac_img)
      #cv2.waitKey()
      th = cv2.adaptiveThreshold(charac_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,5)
      rows, cols = np.nonzero(th)
      charimage = th[(min(rows)):(max(rows)),(min(cols)):(max(cols))]
      shape=charimage.shape
      charimage = cv2.resize(charimage, (30,30))
      charimage = cv2.blur(charimage, (6,6))
# =============================================================================
#       #sobelxchar = cv2.Sobel(charimage,cv2.CV_64F,1,0,ksize=5)
#       #sobelychar = cv2.Sobel(charimage,cv2.CV_64F,0,1,ksize=5)
#       #cv2.normalize(sobelxchar, sobelxchar, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
#       #cv2.normalize(sobelychar, sobelychar, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
#       #sobelchar = np.sqrt(np.power(sobelxchar,2) + np.power(sobelychar,2))
#       #testlist.append(testdata(charac,charimage,shape,sobelchar))
# =============================================================================
      testlist.append(testdata(charac,charimage,shape))      
      
    return testlist
    
      
    raise NotImplementedError

def detection(imgtest):
    """ 
    Use connected component labeling to detect various characters in an test_img.
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
       # TODO: Step 2 : Your Detection code should go here. 
    def find(parent,i):
     if parent[i] == i:
        return i
     if parent[i]!= -i:
         return find(parent,parent[i])
 
    def union(parent,x,y):
     if x == y:
	      return
     
     x_set = find(parent, x)
     y_set = find(parent, y)
     if x_set > y_set:
		    parent[x_set] = y_set
     elif x_set < y_set:
		    parent[y_set] = x_set
    
    labellist=[]  
    parent=[0]
    img=imgtest
    th = cv2.adaptiveThreshold(img,1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,7)
#    cv2.imshow('binaryImage',th)
#    cv2.waitKey()
    #A Bigger Matrix to Eliminate Border Checking In The Raster Scan.
    borderType = cv2.BORDER_CONSTANT
    top = 1
    bottom = 1
    lef = 1
    righ = 1
    val=150
    value = [val, val, val]
    binaryImage = cv2.copyMakeBorder(th, top, bottom, lef, righ, borderType, None, value)
    [imageRows, imageColumns] = np.shape(binaryImage)
    
# Creating A Matrix The Same Dimensions As The Binary Image In Which The Labeling Will Happen.
    labeledImage = np.zeros([imageRows, imageColumns])
    lbImage = labeledImage
    labelCounter = 1
#Over Each Row 
    for r in range(1,imageRows):
   #Over Each Column 
      for c in range(1,imageColumns):
       # If The Pixel Currently Being Scanned Is A Foreground Pixel (1).
       if binaryImage[r, c] == 1:
           # Since 4-Connectivity, Need To Read 2 Labels
           left = int(labeledImage[r, c - 1])
           top = int(labeledImage[r - 1, c])
           # If Left == 0 And Top == 0 -> Create A New Label, And Increment The Label Counter
           if left == 0 and top == 0:
               M = labelCounter
               parent = np.append(parent, [M])
               labelCounter = labelCounter + 1
           # If Left == 0 And Top >= 1 -> Top Label.
           elif left == 0 and top >= 1:
               M = top;
           # If Left >= 1 And Top == 0 ->  Left Label.
           elif left >= 1 and top == 0:
               M = left;
           # If Left >= 1 And Top >= 1 ->The Minimum Of The Two And Copy It
           elif left >= 1 and top >= 1:
               M = min(left,top)
# =============================================================================
               labels = [left,top]               
               for X in labels:                              
                  union(parent,X,M)
# =============================================================================
           labeledImage[r, c] = M
           #print(M)
    uniqu, counts = np.unique(lbImage, return_counts=True) 
    print(len(uniqu))
# =============================================================================
    for r in range(1,imageRows):
      for c in range(1,imageColumns):  
        if binaryImage[r, c]==1:
            lbImage[r, c]=find(parent,int(labeledImage[r, c]))
# =============================================================================
    lbfilter = lbImage
    lbImage=lbImage.astype(int)
    
# =============================================================================
     
    class labeldata:
      def __init__(self,name,count,y,x,h,w,charimage):
         self.name= name
         self.count=count
         self.y=y
         self.x=x
         self.h=h
         self.w=w
         self.charimage=charimage
    uniq, counts = np.unique(lbImage, return_counts=True) 
    print(len(uniq))         
    for i in range(1,len(uniq)):
     find=np.where(lbImage == uniq[i])
     name = i
     count = counts[i]
     x=min(find[0])
     y=min(find[1])
     w=max(find[0])-min(find[0])+1
     h=max(find[1])-min(find[1])+1
     charimageint = lbImage[(min(find[0])):(max(find[0])+1),(min(find[1])):(max(find[1])+1)]
     charimage = lbfilter[(min(find[0])+1):(max(find[0])+1),(min(find[1])+1):(max(find[1])+1)]
# =============================================================================
#      sobelxchar = cv2.Sobel(charimage,cv2.CV_64F,1,0,ksize=5)
#      sobelychar = cv2.Sobel(charimage,cv2.CV_64F,0,1,ksize=5)
#      cv2.normalize(sobelxchar, sobelxchar, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
#      cv2.normalize(sobelychar, sobelychar, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
#      sobelchar = np.sqrt(np.power(sobelxchar,2) + np.power(sobelychar,2))
# =============================================================================
     rows, cols = np.nonzero(charimage)
     charimage[rows, cols] = 255
     
     charimage = cv2.resize(charimage, (35,35))
     charimage = cv2.blur(charimage, (5,5))
     labellist.append(labeldata(name,count,y,x,h,w,charimage))
    print(i)
# =============================================================================
#     for obj in labellist:
#       rows, cols = np.nonzero(obj.charimage)
#       obj.charimage[rows, cols] = 255
# =============================================================================
      #print(obj.name, obj.count, obj.y, obj.x, obj.h, obj.w, sep = '    ')
    return labellist
    raise NotImplementedError

def recognition(LList, TList):

    """ 
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 3 : Your Recognition code should go here.
    recog_list = list()
    #bbox = list()
    for obj1 in LList:
      ssdvallist = list()   
      ssdbool = False
      minval = 7
      bbox = [int(obj1.y),int(obj1.x),int(obj1.h),int(obj1.w)]
      #print(obj1.name)
      #print(bbox)
      i=0
      for obj2 in TList:
          ssdval = np.sum(((obj2.charimage) - (obj1.charimage))**2)/(650250)
          #print(ssdval)
          ssdvallist.append(ssdval)         
         
          if min(ssdvallist)<minval:    
              minval = min(ssdvallist)            
              ssdbool = True
              name1 =  TList[i].name          
          i+=1
           
      if ssdbool == False:
            recog_list.append({"bbox":bbox ,"name":"UNKNOWN"})
      else:
            recog_list.append({"bbox":bbox ,"name":name1})

         
    return recog_list
    
    raise NotImplementedError


def save_results(coordinates, rs_directory):
    """
    Donot modify this code
    """
    results = coordinates
    with open(os.path.join(rs_directory, 'results.json'), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()
    
    characters = []

    all_character_imgs = glob.glob(args.character_folder_path+ "/*")
    
    for each_character in all_character_imgs :
        character_name = "{}".format(os.path.split(each_character)[-1].split('.')[0])
        characters.append([character_name, read_image(each_character, show=False)])

    test_img = read_image(args.test_img)

    results = ocr(test_img, characters)

    save_results(results, args.rs_directory)


if __name__ == "__main__":
    main()
