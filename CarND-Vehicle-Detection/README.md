
# Vehicle-Detection
------


## External Modules for Project 
----


```python
import numpy as np
import cv2
import random
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
import glob
from pathlib import Path
import pickle
import imageio
# imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip
from IPython.display import HTML

```

## Load/Save Helper Functions
----


```python
def load_saved_data(data_path):
    data_path_chk = Path(data_path).is_file()
    if data_path_chk:
        with open(data_path,'rb') as fp:
            saved_data = pickle.load(fp)
        return saved_data
    else:
         raise LookupError("can't find {}".format(data_path))


def save_data_to_path(data_path, data):
    with open(data_path, 'wb') as fp:
        pickle.dump(data, fp)
```

## Data Loading
----
 The data can be seperated as **vehicle** and **non-vehicle** objects, that data files can be downloaded in [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip). After download the zip files, extract the files under directry of **./test_images/**. Since we are using sklearn for the project which doesn't support GPU computation, we will limit the amount of training and testing data to prevent the features extraction and model prediction taking too long. 


```python
# Divide up into vehicle and vehicles
vehicle_images_path = glob.glob('./test_images/vehicles/**/*.png')
non_vehicle_images_path = glob.glob('./test_images/non-vehicles/**/*.png')

vehicle_imgs = []
non_vehicle_imgs = []

read_img = lambda img_path: cv2.imread(img_path)

for image in vehicle_images_path:
    img = read_img(image)
    vehicle_imgs.append(img)
for image in non_vehicle_images_path:
    img = read_img(image)
    non_vehicle_imgs.append(img)

# we only using 3000 vehicle and non-vehicle data 
n_data = 3500

# random select vehicle and non-vehicle objects
random_select_vehicle_imgs = random.sample(vehicle_imgs, n_data)
random_select_non_vehicle_imgs = random.sample(non_vehicle_imgs, n_data)

# create the data and labels for the dataset(traing+testing) 
# since the two catagorties are well seperated, we labeled vehicle as 1 and 0 otherwise
data_set = random_select_vehicle_imgs+random_select_non_vehicle_imgs
data_set_labels = [1 if idx<len(random_select_vehicle_imgs) else 0 for idx in range(len(data_set))]

print("total number of vehicles {}".format(len(vehicle_imgs)))
print("total number of non-vehicles {}".format(len(non_vehicle_imgs)))
print("total number of object {}".format(len(data_set)))
print("total number of object labels {}".format(len(data_set_labels))) 

```

    total number of vehicles 8792
    total number of non-vehicles 8968
    total number of object 7000
    total number of object labels 7000


## Image Features
----
There are three types of image features that we care about in this project. There are spatial (resize), color histrogram, and Oriented Gradient (HOG) 

The **bin_spatial** will return the smaller(32*32) size of the original picture(64*64) in a features vector, this effectly help to cutdown computation 

The **color_histrogram** function basically give out the color(image channel) distribution according to the horizontal axis of the image(i.e how may same color pixel appear in the same bin column). In this case we use 32 bins(model image size:64*64) to repersent the distribution within each channel and then stack those bins within each channel together

The **convert_clr** function is used to convert image from its original color space to a desire color space. In our case we convert the image from RGB space to the **YCrCb** space, because the **YCrCb** can help to extract HOG features 

The main idea of Histogram of Oriented Gradient (HOG) is to compute the gradient information of how color changes within a channel with respect to different directions(i.e # orientations) within each patch(# cells * # pixels/cell)

In our case, we found classiers that we will train later don't have a performance improvement as we increase the number of **orient** ,**cell_per_block** and **pix_per_cell**, but rather increase in processing time. Therefore we decide to keep those parameters as lecture suggested. However, as we stack all the HOG channels and use that for training our classifiers, we did see some improvement in classification. Therefore, we set **hog_channel="ALL"** 





### global parameters for features extraction  
These parameter are mostly for tuning the model 


```python
cspace='YCrCb'
spatial_size=(32, 32)
hist_bins=32
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL" 
```


```python
# color space feature, convert an image to specified color space and size
# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features


# Define a function to compute color histogram features  
def color_hist(img, nbins=32):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def convert_clr(image, cspace='RGB'):
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

    else: feature_image = np.copy(image)     
    return feature_image

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                                  visualise=True, feature_vector=False)
        return features, hog_image
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                       visualise=False, feature_vector=feature_vec)
        return features

#extract HOG of image from specified channel(s)
def extract_hog_features(feature_image, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0):
    
    # Call get_hog_features() with vis=False, feature_vec=True
    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(feature_image.shape[2]):
            hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                orient, pix_per_cell, cell_per_block, 
                                vis=False, feature_vec=True))
        hog_features = np.ravel(hog_features)        
    else:
        hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                    pix_per_cell, cell_per_block, vis=False, feature_vec=True)
   
    return np.array(hog_features)
```

### tesing HOG features function
we tested the HOG features of both vehicle and non-vehicle objects for all channel


```python
%matplotlib inline

selected_vehicle = random.choice(vehicle_imgs)
selected_non_vehicle = random.choice(non_vehicle_imgs)

def test_HOG(img):
    new_img = convert_clr(img)
    _, hog_img_ch0= get_hog_features(new_img[:,:,0], orient,pix_per_cell,cell_per_block, True)
    _, hog_img_ch1= get_hog_features(new_img[:,:,1], orient,pix_per_cell,cell_per_block, True)
    _, hog_img_ch2= get_hog_features(new_img[:,:,2], orient,pix_per_cell,cell_per_block, True)

    return [
            new_img,
            hog_img_ch0,
            hog_img_ch1,
            hog_img_ch2
            ]


output_imgs = [test_HOG(selected_vehicle), test_HOG(selected_non_vehicle)] 
ouput_labels = ["oringinal image", "HOG-0", "HOG-1", "HOG-2"]


fig = plt.figure()
for idx, cat in enumerate(output_imgs):
    for idx2, img in enumerate(cat):
        fig.add_subplot(idx+1,len(cat), idx2+1)
        plt.axis('off')
        plt.title(ouput_labels[idx2])
        plt.imshow(img)
        plt.subplots_adjust(left=0.1, right=0.9, top=1.2, bottom=0)
```


![png](output_12_0.png)


## The model pipeline
---
we create the model pipeline using the concatenation(horizontal) of above features, then stacking(vertical) them up for all the training examples, then utilize **StandardScaler** from **sklearn** to nomalize the incoming data

During training, the model pipeline will generate a normalization base and the updated(after normalization) features of all the training examples. 

During testing, the testing example can be normailized by the normalization base that provided from training via function's parameters passing. 

we also include the argumentation method for training the model. The main raeson is even the impact of image argumentation is insignificant to the svc training, but linear SVC can not provide a good enough model for classification. Therefore, we also trained a neural network to jointly predict the classes, and one benifit for image argumentation is that it can help to prevent overfitting while training the neural network.  


```python
def img_brightness_argumentation(img):
    img = np.asarray(img)
    hsv_img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    random_bright = .35+np.random.uniform()
    hsv_img[:,:,2] = hsv_img[:,:,2]*random_bright
    return cv2.cvtColor(hsv_img,cv2.COLOR_HSV2RGB)


def img_bluring_argumentation(img):
    img = np.asarray(img)

    # choose a random bluring index
    rand_blur_idx = random.choice([3,5,7])

    # decide whether we should use standard bluring on this step
    img = (cv2.blur(img,(rand_blur_idx,rand_blur_idx))
           if random.randint(0,1)==1 else img)

    # decide whether we should use Gaussian bluring on this step
    img = (cv2.GaussianBlur(img,(rand_blur_idx,rand_blur_idx),0)
           if random.randint(0,1)==1 else img)

    # decide whether we should use median bluring on this step
    img = (cv2.medianBlur(img,rand_blur_idx)
           if random.randint(0,1)==1 else img)
    return img


# this function is mainly used by training to generate various
# combination of image argumentation
def img_argumentation(img):
    # decide whether we should use brightness arguementation on this step
    arg_img = (img_brightness_argumentation(img)
               if random.randint(0,1)==1 else img)

    # decide whether we should use any bluring arguementation on this step
    arg_img = (img_bluring_argumentation(arg_img)
               if random.randint(0,1)==1 else arg_img)

    return arg_img
```


```python
def model_pipeline(data, ttl_normalized_base=None, allow_fit=True):
    
    ttl_fts = None

    if not (ttl_normalized_base):
        ttl_normalized_base = StandardScaler()
        
    for idx, obj in enumerate(data):
    
        obj = img_argumentation(obj) if random.choice(range(10))<=1 else obj
        
        feature_image = convert_clr(obj, cspace)
                        
        spatial_ft = bin_spatial(feature_image, size=spatial_size)
        
        hist_ft = color_hist(feature_image, nbins=hist_bins)
        
        HOG_ft = extract_hog_features(feature_image, orient, pix_per_cell, cell_per_block, hog_channel)
                
        ttl_ft = np.hstack((spatial_ft, hist_ft, HOG_ft))
        
        ttl_fts = np.vstack((ttl_fts,ttl_ft)).astype(np.float64) if idx>0 else ttl_ft.astype(np.float64)  
    
    if allow_fit:
        
        ttl_normalized_base = ttl_normalized_base.fit(ttl_fts)


    ttl_normalized_features = ttl_normalized_base.transform(ttl_fts)
    
    return ttl_normalized_base, ttl_normalized_features
```

### load/save normalization base and nomalized features


```python
train_data_path = './train_data'
train_labels_path = './train_lables'
test_data_path = './test_data'
test_labels_path = './test_lables'
train_normalized_base_path = './train_normalized_base'
train_normalized_features_path = './train_normalized_features'
test_normalized_base_path = './test_normalized_base'
test_normalized_features_path = './test_normalized_features'

try:
    X_train,y_train = load_saved_data(train_data_path), load_saved_data(train_labels_path)
    X_test,y_test = load_saved_data(test_data_path), load_saved_data(test_labels_path)
    train_normalized_base = load_saved_data(train_normalized_base_path)
    train_normalized_features = load_saved_data(train_normalized_features_path)
    test_normalized_base = load_saved_data(test_normalized_base_path)
    test_normalized_features = load_saved_data(test_normalized_features_path)

    
except:
    all_data = X_train, X_test, y_train, y_test = train_test_split(
    data_set, data_set_labels, test_size=0.15, random_state=87)


    data_paths = [train_data_path, test_data_path, train_labels_path, test_labels_path]
    for path, data in zip(data_paths, all_data):
        save_data_to_path(path, data)
    
    
    train_normalized_base, train_normalized_features = model_pipeline(X_train)
    test_normalized_base, test_normalized_features = model_pipeline(X_test, train_normalized_base, False)
    save_pipelined_data = [train_normalized_base, train_normalized_features,
                           test_normalized_base, test_normalized_features]
   
    save_pipelined_paths = [train_normalized_base_path, train_normalized_features_path,
                            test_normalized_base_path, test_normalized_features_path]
    
    for path, data in zip(save_pipelined_paths, save_pipelined_data):
        save_data_to_path(path, data)

print("total number of training objects {}".format(len(X_train)))
print("total number of testing objects {}".format(len(X_test)))
```

    total number of training objects 5100
    total number of testing objects 900


## Training Model
-----
In here, we train a linear SVC together with a 3 layers Neural Network(nn) to jointly classified vehicle and non-vehicle objects. 



```python
%matplotlib inline

svc_path = './svc_model'
nn_path = './nn_model'

t=time.time()

try:
    svc = load_saved_data(svc_path)
    nn = load_saved_data(nn_path)
except:
    svc = LinearSVC(C=5e-2, loss='hinge')
    svc.fit(train_normalized_features, y_train)
    nn = MLPClassifier(hidden_layer_sizes=(256,128,16))
    nn.fit(train_normalized_features, y_train)    
    save_data_to_path(svc_path, svc)
    save_data_to_path(nn_path, nn)


t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC nad NN...')

print('Test Accuracy of SVC = ', round(svc.score(test_normalized_features, y_test), 4))
print('Test Accuracy of Neural Network = ', round(nn.score(test_normalized_features, y_test), 4))

# Check the prediction time for a single sample
t=time.time()
n_predict = 10
n_prediction=svc.predict(test_normalized_features[0:n_predict])
n_prediction_2=nn.predict(test_normalized_features[0:n_predict])

n_truth=y_test[0:n_predict]

print('SVC predicts: ',n_prediction )
print('Neural Network predicts: ',n_prediction_2 )
print('For these',n_predict, 'labels: ', n_truth)
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

# lambda function that used for identify object type from number 
eval_sign = lambda num: "vehicle" if num==1 else "non-vehicle"

fig = plt.figure()
for idx in range(n_predict):
    fig.add_subplot((idx)/5+1,5, idx+1)
    plt.axis('off')
    plt.title("svc predict: {},\nnn  predict: {},\nactual: {}".format(eval_sign(n_prediction[idx]),
                                                                         eval_sign(n_prediction_2[idx]),
                                                                         eval_sign(n_truth[idx])),fontsize=6)
    plt.imshow(X_test[idx])
    plt.subplots_adjust(left=0.05, right=0.9, top=1.2, bottom=0)


```

    0.05 Seconds to train SVC nad NN...
    Test Accuracy of SVC =  0.9878
    Test Accuracy of Neural Network =  0.9922
    SVC predicts:  [1 1 1 0 1 0 1 1 1 1]
    Neural Network predicts:  [1 1 1 0 1 0 1 1 1 1]
    For these 10 labels:  [1, 1, 1, 0, 1, 0, 1, 1, 1, 1]
    0.03125 Seconds to predict 10 labels with SVC



![png](output_19_1.png)


## Object indentification functions

Since our classifiers can only idetify which classes an image belong to, by using the corresponding feature vector from image. If an image contains mutiple objects of interest. One can use sliding window technique to take patches of the image and feed those patches to the classifier, to identify what class each patch belongs to. 

pros and cons for sliding window technique
**Pro**: easy to implement
**Con**: take a long time to classify every patch within the image, if the patch size is small, but image size is big. 

Therefore, we select only a portion of the image to apply the sliding window technique. In our case we choose the image region from **ystart=350** and **ystop=656** in the y-axis of the image. We did this selection because objects that we want to classified within the image mostly occour in that y value range 

There are three main functions in here

**find_cars**: scale a region of interest(area that vehicle/non-vehicle appear offen) and then apply the features extraction as above, by going thru the region using sliding windows and classify each window, one can draw boex and record the coordinates of that box if a window has been classified as vehicle

**heatmap**: due to the fact that find_car might re-draw many duplicate region becasue of the silid window technique, one can use heatmap(add 1 to an area) on the identified area from coordinates that mentioned above then apply a threshold to filter out a region

**draw_box**: draw a box on the region that the heatmap identified



```python
def find_cars(img, ystart, ystop, scale, model1, model2, X_scaler,
              orient=orient, pix_per_cell=pix_per_cell, 
              cell_per_block=cell_per_block, spatial_size=spatial_size,
              hist_bins=hist_bins):
    
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    img_tosearch = img[ystart:ystop,:,:]  
    bounding_boxes = []
    ctrans_tosearch = convert_clr(img_tosearch, 'YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    cells_per_step = 2 # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)))    

            model1_prediction = model1.predict(test_features)
            model2_prediction = model2.predict(test_features)
      
            if model1_prediction==1 and model2_prediction==1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                bounding_boxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))

    return draw_img,bounding_boxes



def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img


```

## Image/Video Process Pipeline
----
we tried multiple value of **scale** and **threshold**, and we found the classifiers perform the best while we maintain the image scale. And the **threshold=4** help us to reduce false postive within the heatmap while maintain objects that we want to detect   


```python
def image_process_pipeline(image, model1=svc, model2=nn,
                           model_scaler=train_normalized_base):
    ystart = 350
    ystop = 656
    scale = 1
    threshold = 4
    
    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    int_draw_pic, bounding_boxes = find_cars(image, ystart, ystop, scale, model1, model2, model_scaler)
    
    heat = add_heat(heat, bounding_boxes)
    
    heat = apply_threshold(heat, threshold)
    
    heatmap = np.clip(heat, 0, 255)

    labels = label(heatmap)
    
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    
    return draw_img, heatmap, int_draw_pic

def video_process_pipeline(image):
    img, _, _ =image_process_pipeline(image)
    return img
```

### test image pipeline 


```python
%matplotlib inline
test_images_path = glob.glob('./test_images/*.jpg')
titles = ["oringinal","initial boxes", "heat map", "detected boxes"]

for ix, path in enumerate(test_images_path):
    image = read_img(path)
    draw_img, heatmap,int_draw_pic = image_process_pipeline(image)
    imgs = [image, int_draw_pic, heatmap, draw_img]
    fig = plt.figure()
    for ix2, img in enumerate(imgs):
        fig.add_subplot(ix+1, 4, ix2+1)
        plt.axis('off')
        plt.title(titles[ix2])
        plt.imshow(img)
        plt.subplots_adjust(left=0.1, right=0.9, top=1.2, bottom=0)
```


![png](output_25_0.png)



![png](output_25_1.png)



![png](output_25_2.png)



![png](output_25_3.png)



![png](output_25_4.png)



![png](output_25_5.png)


### test video pipeline


```python
processed_output = 'output2.mp4'
clip1 = VideoFileClip("project_video.mp4")
processed_clip = clip1.fl_image(video_process_pipeline) #NOTE: this function expects color images!!
%time processed_clip.write_videofile(processed_output, audio=False)
```

## Conclusions:
----
This project we untilzed multiple feature vectors, linear SVM&NN classifiers together with some image manipulation techniques(sliding window & heatmap) to identify objects within an image.

The pipeline can identify objects within a image most of the time, however, it also generates some false postives, which might due to the limited trainig size, and overfitting of the models. Also, due to the nature of we applying threshold on an image to filter false positive, the ground truth boxes usuually cannot correctly contain the objects it identiify

While the project is easy to understand and implemented, it suffers from many drawbacks such as problem that mention above, togehter with the fact that the whole pipeline cannot process video stream in real time, which might be essential for self driving vehicles. In order to build a better pipeline, One should also consider techniques that utilize deep nerual network(e.g: SSD, YOLO, etc) 
