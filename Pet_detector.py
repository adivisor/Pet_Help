import os
import cv2
import numpy as np
# from picamera.array import PiRGBArray
# from picamera import PiCamera
import tensorflow as tf
import argparse
import sys

# Twilio-Setup
from twilio.rest import Client

twilio_sid = os.environ['TWILIO_ACCOUNT_SID']
auth_token = os.environ['TWILIO_AUTH_TOKEN']
phone_no = os.environ['MY_DIGITS']
twilio_number = os.environ['TWILIO_DIGITS']
client = Client(twilio_sid,auth_token)

cam_width = 1920
cam_height = 1080

camera_type = 'picamera'
parser = argparse.ArgumentParser()
parser.add_argument('--usbcam', help='Use a USB webcam instead of picamera',
                    action='store_true')
args = parser.parse_args()
if args.usbcam:
    camera_type = 'usb'


#### Initialize TensorFlow model ####
sys.path.append('..')
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
CWD_PATH = os.getcwd()

PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_label_map.pbtxt')
NUM_CLASSES = 90


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detect_graph = tf.Graph()
with detect_graph.as_default():
    graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(graph_def, name='')

    sess = tf.compat.v1.Session(graph=detect_graph)



image_tensor = detect_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
detect_boxes = detect_graph.get_tensor_by_name('detect_boxes:0')

detect_scores = detect_graph.get_tensor_by_name('detect_scores:0')
detect_classes = detect_graph.get_tensor_by_name('detect_classes:0')

num_detections = detect_graph.get_tensor_by_name('num_detections:0')


# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX
TopLeft_in = (int(cam_width*0.05),int(cam_height*0.35))
TopLeft_out = (int(cam_width*0.28),int(cam_height*0.25))
BottomRight_in = (int(cam_width*0.45),int(cam_height-5))
BottomRight_out = (int(cam_width*0.8),int(cam_height*.85))

# Initialize control variables used for pet detector
pet_inside = False
detected_outside = False
inside_counter = 0
outside_counter = 0
pause = 0
pause_counter = 0

#### Pet detection function ####
def pet_detector(frame):

    global pet_inside, detected_outside
    global inside_counter, outside_counter
    global pause, pause_counter

    frame_exp = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detect_boxes, detect_scores, detect_classes, num_detections],
        feed_dict={image_tensor: frame_exp})

    # Draw the results of the detection (aka 'visulaize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.40)

    # Draw boxes defining "outside" and "inside" locations.
    cv2.rectangle(frame,TopLeft_out,BottomRight_out,(255,20,20),3)
    cv2.putText(frame,"Outside box",(TopLeft_out[0]+10,TopLeft_out[1]-10),font,1,(255,20,255),3,cv2.LINE_AA)
    cv2.rectangle(frame,TopLeft_in,BottomRight_in,(20,20,255),3)
    cv2.putText(frame,"Inside box",(TopLeft_in[0]+10,TopLeft_in[1]-10),font,1,(20,255,255),3,cv2.LINE_AA)
    
    if (((int(classes[0][0]) == 18) or (int(classes[0][0] == 17) or (int(classes[0][0]) == 88))) and (pause == 0)):
        x = int(((boxes[0][0][1]+boxes[0][0][3])/2)*cam_width)
        y = int(((boxes[0][0][0]+boxes[0][0][2])/2)*cam_height)

        cv2.circle(frame,(x,y), 5, (75,13,180), -1)

        if ((x > TopLeft_in[0]) and (x < BottomRight_in[0]) and (y > TopLeft_in[1]) and (y < BottomRight_in[1])):
            inside_counter = inside_counter + 1

        if ((x > TopLeft_out[0]) and (x < BottomRight_out[0]) and (y > TopLeft_out[1]) and (y < BottomRight_out[1])):
            outside_counter = outside_counter + 1


    if inside_counter > 10:
        pet_inside = True
        message = client.messages.create(
            body = 'Your pet wants outside!',
            from_=twilio_number,
            to=phone_no
            )
        inside_counter = 0
        outside_counter = 0
        # Pause pet detection by setting "pause" flag
        pause = 1

    # Set the detected outside flag and send a text to the phone 
    if outside_counter > 10:
        detected_outside = True
        message = client.messages.create(
            body = 'Your pet wants inside!',
            from_=twilio_number,
            to=phone_no
            )
        inside_counter = 0
        outside_counter = 0
        pause = 1

    if pause == 1:
            
        if pet_inside == True:
            cv2.putText(frame,'Pet wants outside!',(int(cam_width*.1),int(cam_height*.5)),font,3,(0,0,0),7,cv2.LINE_AA)
            cv2.putText(frame,'Pet wants outside!',(int(cam_width*.1),int(cam_height*.5)),font,3,(95,176,23),5,cv2.LINE_AA)

        if pet_outside == True:
            cv2.putText(frame,'Pet wants inside!',(int(cam_width*.1),int(cam_height*.5)),font,3,(0,0,0),7,cv2.LINE_AA)
            cv2.putText(frame,'Pet wants inside!',(int(cam_width*.1),int(cam_height*.5)),font,3,(95,176,23),5,cv2.LINE_AA)

        # Increment pause counter until it reaches 30 (for a framerate of 1.5 FPS, this is about 20 seconds),
        # then unpause the application (set pause flag to 0).
        pause_counter = pause_counter + 1
        if pause_counter > 30:
            pause = 0
            pause_counter = 0
            pet_inside = False
            pet_outside = False


    


if camera_type == 'picamera':
    camera.resolution = (cam_width,cam_height)
    rawCapture = PiRGBArray(camera, size=(cam_width,cam_height))
    rawCapture.truncate(0)

    # Continuously capture frames and perform object detection on them
    for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):

        t1 = cv2.getTickCount()
        
        frame = frame1.array
        frame.setflags(write=1)

        frame = pet_detector(frame)

        cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)

        cv2.imshow('Object detector', frame)

        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1

        if cv2.waitKey(1) == ord('q'):
            break

        rawCapture.truncate(0)

    camera.close()

### USB webcam ###
    
elif camera_type == 'usb':
    ret = camera.set(3,cam_width)
    ret = camera.set(4,cam_height)

    # Continuously capture frames and perform object detection on them
    while(True):

        t1 = cv2.getTickCount()
        ret, frame = camera.read()
        frame = pet_detector(frame)

        cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)

        cv2.imshow('Object detector', frame)

        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1

        if cv2.waitKey(1) == ord('q'):
            break

    camera.release()
        
cv2.destroyAllWindows()