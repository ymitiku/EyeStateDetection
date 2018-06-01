import cv2
import numpy as np
import dlib
import os
from keras.models import model_from_json
import argparse
import json


def get_cmd_args():
    """Gets cmd arguments        
    """
    parser = argparse.ArgumentParser()
    with open("config.json") as f:
        data = json.load(f)
    parser.add_argument("--json_path",default="models/model.json",type=str)
    parser.add_argument("--weights",default="models/model.h5",type=str)
    parser.add_argument("--process",default="webcam",type=str)
    parser.add_argument("--path",default="-1",type=str)
    #parser.add_argument("--image_size",default=24, type=int)
    parser.add_argument('--image_size',default=(data["facial_image_size"]["image_height"], data["facial_image_size"]["image_width"]), nargs='+', type=int)

    args = parser.parse_args()
    return args

def get_dlib_points(img,predictor,rectangle):
    """Extracts dlib key points from face image
    parameters
    ----------
    img : numpy.ndarray
        Grayscale face image
    predictor : dlib.shape_predictor
        shape predictor which is used to localize key points from face image
    rectangle : dlib.rectangle
        face bounding box inside image
    Returns
    -------
    numpy.ndarray
        dlib key points of the face inside rectangle.
    """

    shape = predictor(img,rectangle)
    dlib_points = np.zeros((68,2))
    for i,part in enumerate(shape.parts()):
        dlib_points[i] = [part.x,part.y]
    return dlib_points


def distance_between(v1,v2):
    """Calculates euclidean distance between two vectors. 
    If one of the arguments is matrix then the output is calculated for each row
    of that matrix.

    Parameters
    ----------
    v1 : numpy.ndarray
        First vector
    v2 : numpy.ndarray
        Second vector
    
    Returns:
    --------
    numpy.ndarray
        Matrix if one of the arguments is matrix and vector if both arguments are vectors.
    """

    
    diff = v2 - v1
    diff_squared = np.square(diff)
    dist_squared = diff_squared.sum(axis=1) 
    dists = np.sqrt(dist_squared)
    return dists

def angles_between(v1,v2):
    """Calculates angle between two point vectors. 
    Parameters
    ----------
    v1 : numpy.ndarray
        First vector
    v2 : numpy.ndarray
        Second vector
    
    Returns:
    --------
    numpy.ndarray
        Vector if one of the arguments is matrix and scalar if both arguments are vectors.
    """
    dot_prod = (v1 * v2).sum(axis=1)
    v1_norm = np.linalg.norm(v1,axis=1)
    v2_norm = np.linalg.norm(v2,axis=1)
    

    cosine_of_angle = (dot_prod/(v1_norm * v2_norm)).reshape(11,1)

    angles = np.arccos(np.clip(cosine_of_angle,-1,1))

    return angles

def get_right_key_points(key_points):
    """Extract dlib key points from right eye region including eye brow region.
    Parameters
    ----------
    key_points : numpy.ndarray
        Dlib face key points
    Returns:
        dlib key points of right eye region
    """

    output = np.zeros((11,2))
    output[0:5] = key_points[17:22]
    output[5:11] = key_points[36:42]
    return output

def get_left_key_points(key_points):
    """Extract dlib key points from left eye region including eye brow region.
    Parameters
    ----------
    key_points : numpy.ndarray
        Dlib face key points
    Returns:
        dlib key points of left eye region
    """
    output = np.zeros((11,2))
    output[0:5] = key_points[22:27]
    output[5:11] = key_points[42:48]
    return output

def get_attributes_wrt_local_frame(face_image,key_points_11,image_shape):
    """Extracts eye image, key points of the eye region with respect 
    face eye image, angles and distances between centroid of key point of eye  and
    other key points of the eye.
    Parameters
    ----------
    face_image : numpy.ndarray
        Image of the face
    key_points_11 : numpy.ndarray
        Eleven key points of the eye including eyebrow region.
    image_shape : tuple
        Shape of the output eye image
    
    Returns
    -------
    eye_image : numpy.ndarray
        Image of the eye region
    key_points_11 : numpy.ndarray
        Eleven key points translated to eye image frame
    dists : numpy.ndarray
        Distances of each 11 key points from centeroid of all 11 key points
    angles : numpy.ndarray
        Angles between each 11 key points from centeroid 
    
    """

    face_image_shape = face_image.shape
    top_left = key_points_11.min(axis=0)
    bottom_right = key_points_11.max(axis=0)

    # bound the coordinate system inside eye image
    bottom_right[0] = min(face_image_shape[1],bottom_right[0])
    bottom_right[1] = min(face_image_shape[0],bottom_right[1]+5)
    top_left[0] = max(0,top_left[0])
    top_left[1] = max(0,top_left[1])

    # crop the eye
    top_left = top_left.astype(np.uint8)
    bottom_right = bottom_right.astype(np.uint8)
    eye_image = face_image[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]]

    # translate the eye key points from face image frame to eye image frame
    key_points_11 = key_points_11 - top_left
    key_points_11 += np.finfo(float).eps

    # horizontal scale to resize image
    scale_h = image_shape[1]/float(eye_image.shape[1])

    # vertical scale to resize image
    scale_v = image_shape[0]/float(eye_image.shape[0])

    # resize left eye image to network input size
    eye_image = cv2.resize(eye_image,(image_shape[0],image_shape[1]))

    # scale left key points proportional with respect to left eye image resize scale
    scale = np.array([[scale_h,scale_v]])
    key_points_11 = key_points_11 * scale 

    # calculate centroid of left eye key points 
    centroid = np.array([key_points_11.mean(axis=0)])

    # calculate distances from  centroid to each left eye key points
    dists = distance_between(key_points_11,centroid)

    # calculate angles between centroid point vector and left eye key points vectors
    angles = angles_between(key_points_11,centroid)
    return eye_image, key_points_11,dists,angles

def get_left_eye_attributes(face_image,predictor,image_shape):
    """Extracts eye image, key points, distance of each key points
    from centroid of the key points and angles between centroid and
    each key points of left eye.
    
    Parameters
    ----------
    face_image : numpy.ndarray
        Image of the face
    predictor : dlib.shape_predictor
        Dlib Shape predictor to extract key points
    image_shape : tuple
        The output eye image shape
    Returns
    -------
    eye_image : numpy.ndarray
        Image of the eye region
    key_points_11 : numpy.ndarray
        Eleven key points translated to eye image frame
    dists : numpy.ndarray
        Distances of each 11 key points from centeroid of all 11 key points
    angles : numpy.ndarray
        Angles between each 11 key points from centeroid 
    
    """
    face_image_shape = face_image.shape
    face_rect = dlib.rectangle(0,0,face_image_shape[1],face_image_shape[0])
    kps = get_dlib_points(face_image,predictor,face_rect)
    # Get key points of the eye and eyebrow

    key_points_11 = get_left_key_points(kps)
    
    eye_image,key_points_11,dists,angles = get_attributes_wrt_local_frame(face_image,key_points_11,image_shape)

    return eye_image,key_points_11,dists,angles

def get_right_eye_attributes(face_image,predictor,image_shape):
    """Extracts eye image, key points, distance of each key points
    from centroid of the key points and angles between centroid and
    each key points of right eye.
    
    Parameters
    ----------
    face_image : numpy.ndarray
        Image of the face
    predictor : dlib.shape_predictor
        Dlib Shape predictor to extract key points
    image_shape : tuple
        The output eye image shape
    Returns
    -------
    eye_image : numpy.ndarray
        Image of the eye region
    key_points_11 : numpy.ndarray
        Eleven key points translated to eye image frame
    dists : numpy.ndarray
        Distances of each 11 key points from centeroid of all 11 key points
    angles : numpy.ndarray
        Angles between each 11 key points from centeroid 
    
    """
    face_image_shape = face_image.shape
    face_rect = dlib.rectangle(0,0,face_image_shape[1],face_image_shape[0])
    kps = get_dlib_points(face_image,predictor,face_rect)
    # Get key points of the eye and eyebrow

    key_points_11 = get_right_key_points(kps)
    
    eye_image,key_points_11,dists,angles = get_attributes_wrt_local_frame(face_image,key_points_11,image_shape)

    return eye_image,key_points_11,dists,angles

def load_model(json_path,weights_path):
    """ Loads keras model 
    Parameters
    ----------
    json_path : str
        Path to json file of the model
    weights_Path : str
        Path to weights of the model
    Returns 
    keras.model.Model 
        Model with weights loadsed
    """
    assert os.path.exists(json_path),"json file path "+str(json_path)+" does not exist"
    assert os.path.exists(json_path),"weights file path "+str(weights_path)+" does not exist"

    with open(json_path,"r") as json_file:
        model_json = json_file.read()
        model = model_from_json(model_json)
        model.load_weights(weights_path)
        return model

def webcam_demo(json_path,weights_path):
    """ Webcam demo
    Parametres
    ----------
    json_path : str
        Path to json file of the model
    weights_Path : str
        Path to weights of the model
    """  
    model = load_model(json_path,weights_path)
    process_video(model,-1)

def video_demo(json_path,weights_path,video_path):
    """Video demo
    Parameters
    ----------
    json_path : str
        Path to json file of the model
    weights_Path : str
        Path to weights of the model
    video_path : str
        Path to video
    """
    model = load_model(json_path,weights_path)
    process_video(model,video_path)

def image_demo(json_path,weights_path,image_path,image_height,image_width):
    print(image_height)
    print(image_width)
    img = cv2.imread(image_path)
    if not img is None:
        
        model = load_model(json_path,weights_path)
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        print (img.shape)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)       #convert the image to grey scale
        faces = detector(gray)
        for i,face in enumerate(faces):
            face_img = gray[
                    max(0,face.top()):min(gray.shape[0],face.bottom()),
                    max(0,face.left()):min(gray.shape[1],face.right())
            ]
            cv2.rectangle(img,(face.left(),face.top()),(face.right(),face.bottom()),color=(255,0,0),thickness=2)
            face_img = cv2.resize(face_img,(100,100))
            l_i,lkp,ld,la = get_left_eye_attributes(face_img,predictor,(image_height,image_width,1))
            r_i,rkp,rd,ra = get_right_eye_attributes(face_img,predictor,(image_height,image_width,1))
            
            cv2.imshow("Left eye: ",l_i)
            cv2.imshow("Right eye: ",r_i)

            l_i = l_i.reshape(-1,image_height,image_width,1).astype(np.float32)/255
            r_i = r_i.reshape(-1,image_height,image_width,1).astype(np.float32)/255

            lkp = np.expand_dims(lkp,1).astype(np.float32)/image_height
            ld = np.expand_dims(ld,1).astype(np.float32)/image_width
            la = np.expand_dims(la,1).astype(np.float32)/np.pi

            rkp = np.expand_dims(rkp,1).astype(np.float32)/image_height
            rd = np.expand_dims(rd,1).astype(np.float32)/image_width
            ra = np.expand_dims(ra,1).astype(np.float32)/np.pi

            lkp = lkp.reshape(-1,1,11,2)
            ld = ld.reshape(-1,1,11,1)
            la = la.reshape(-1,1,11,1)

            rkp = rkp.reshape(-1,1,11,2)
            rd = rd.reshape(-1,1,11,1)
            ra = ra.reshape(-1,1,11,1)

            left_prediction = model.predict([l_i,lkp,ld,la])[0]
            right_prediction = model.predict([r_i,rkp,rd,ra])[0]
            
            left_arg_max = np.argmax(left_prediction)
            right_arg_max = np.argmax(right_prediction)

            if left_arg_max == 0:
                left_text = "Left eye Closed"
            else:
                left_text = "Left eye Opened"
            if right_arg_max == 0:
                right_text = "Right eye Closed"
            else:
                right_text = "Right eye Opened"
            cv2.putText(img,left_text,(face.left()+10,face.top()+10), cv2.FONT_HERSHEY_DUPLEX, 0.4,color=(0,0,255),thickness=1)
            cv2.putText(img,right_text,(face.left()+10,face.top()+30), cv2.FONT_HERSHEY_DUPLEX, 0.4,color=(0,0,255),thickness=1)
        cv2.imshow("Image",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print ("Unable to read image from",image_path)

def process_video(model,video_path):
    """processes either webcam or video.
    Parametres
    ----------
    model : keras.models.Model
        model to be used for prediction
    video_path : str or int
        video path if it is str or webcam if it is int 
    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        _, frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        for i,face in enumerate(faces):
            face_img = gray[
                    max(0,face.top()):min(gray.shape[0],face.bottom()),
                    max(0,face.left()):min(gray.shape[1],face.right())
            ]
            cv2.rectangle(frame,(face.left(),face.top()),(face.right(),face.bottom()),color=(255,0,0),thickness=2)
            face_img = cv2.resize(face_img,(100,100))
            l_i,lkp,ld,la = get_left_eye_attributes(face_img,predictor,(image_size,image_size,1))
            r_i,rkp,rd,ra = get_right_eye_attributes(face_img,predictor,(image_size,image_size,1))
            
            cv2.imshow("Left eye: ",l_i)
            # for kp in lkp:
            #     cv2.circle(l_i,(kp[0],kp[1]),1,(255,255,0))
            cv2.imshow("Right eye: ",r_i)
            l_i = l_i.reshape(-1,args.image_size,args.image_size,1).astype(np.float32)/255
            r_i = r_i.reshape(-1,args.image_size,args.image_size,1).astype(np.float32)/255

            lkp = np.expand_dims(lkp,1).astype(np.float32)/image_size
            ld = np.expand_dims(ld,1).astype(np.float32)/image_size
            la = np.expand_dims(la,1).astype(np.float32)/np.pi

            rkp = np.expand_dims(rkp,1).astype(np.float32)/image_size
            rd = np.expand_dims(rd,1).astype(np.float32)/image_size
            ra = np.expand_dims(ra,1).astype(np.float32)/np.pi

            lkp = lkp.reshape(-1,1,11,2)
            ld = ld.reshape(-1,1,11,1)
            la = la.reshape(-1,1,11,1)

            rkp = rkp.reshape(-1,1,11,2)
            rd = rd.reshape(-1,1,11,1)
            ra = ra.reshape(-1,1,11,1)

            left_prediction = model.predict([l_i,lkp,ld,la])[0]
            right_prediction = model.predict([r_i,rkp,rd,ra])[0]
            
            left_arg_max = np.argmax(left_prediction)
            right_arg_max = np.argmax(right_prediction)

            if left_arg_max ==0:
                left_text = "Left eye Closed"
            else:
                left_text = "Left eye Opened"
            if right_arg_max ==0:
                right_text = "Right eye Closed"
            else:
                right_text = "Right eye Opened"
            cv2.putText(frame,left_text,(face.left()+10,face.top()+10), cv2.FONT_HERSHEY_DUPLEX, 0.4,color=(0,0,255),thickness=1)
            cv2.putText(frame,right_text,(face.left()+10,face.top()+30), cv2.FONT_HERSHEY_DUPLEX, 0.4,color=(0,0,255),thickness=1)
        cv2.imshow("Frame",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()
