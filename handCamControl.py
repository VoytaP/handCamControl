import bpy
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

SCALE = 0.01
PATH = 'C:/Tvorba/blenderPokusy/addonPokus/hand_landmarker.task'
MAXTIMESTAMP = 500

# Create an HandLandmarker object
BaseOptions = python.BaseOptions(model_asset_path=PATH)
VisionRunningMode = vision.RunningMode
options = vision.HandLandmarkerOptions(base_options=BaseOptions, num_hands=1, running_mode=VisionRunningMode.VIDEO)
detector = vision.HandLandmarker.create_from_options(options)

# Create function to control hand in Blender
cap =cv2.VideoCapture(0)
timeStamp = 0
def controlHand():
    global timeStamp
    
    ret, frame = cap.read()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detection_result = detector.detect_for_video(mp_image, timeStamp)
    hand_landmarks_list = detection_result.hand_landmarks

    img = cv2.cvtColor(mp_image.numpy_view(), cv2.COLOR_RGB2BGR)
    imgSize = np.shape(img)
    for oneHand_landmarks in hand_landmarks_list:
        for index, lmark in enumerate(oneHand_landmarks):
            xVal = SCALE * imgSize[1] * lmark.x
            yVal = SCALE * imgSize[0] * lmark.y
            zVal = SCALE * imgSize[0] * lmark.z
            LMName = 'LM' + str(index)
            LMObject = bpy.data.objects[LMName]
            LMObject.location = [xVal, yVal, zVal]

    print(timeStamp)
    timeStamp += 1 
    if timeStamp == MAXTIMESTAMP:
        return None
    return 0.1

bpy.app.timers.register(controlHand)
