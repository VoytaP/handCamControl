import bpy
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

SCALE = 0.01
PATH = 'C:/Tvorba/handCamControl/hand_landmarker.task'

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
    return 0.1

# Create Blender UI
class StartHandCamOperator(bpy.types.Operator):
    """Start handCamControl"""
    bl_idname = "hand.start_hand_cam"
    bl_label = "startHandCam Operator"

    def execute(self, context):
        bpy.app.timers.register(controlHand)
        return {'FINISHED'}
        
class StopHandCamOperator(bpy.types.Operator):
    """Stop handCamControl"""
    bl_idname = "hand.stop_hand_cam"
    bl_label = "stopHandCam Operator"

    def execute(self, context):
        bpy.app.timers.unregister(controlHand)
        return {'FINISHED'}
    
class HandCamPanel(bpy.types.Panel):
    """Creates a Panel to start/stop handCamControl"""
    
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "HandCamControl"
    bl_label = "HandCamControl"

    def draw(self, context):
        """layout of the panel"""
        running = bpy.app.timers.is_registered(controlHand)
        row = self.layout.row()
        row.operator("hand.start_hand_cam", text="Start")
        row.enabled = not running
        row = self.layout.row()
        row.operator("hand.stop_hand_cam", text="Stop")
        row.enabled = running
        
def register():
    bpy.utils.register_class(StartHandCamOperator)
    bpy.utils.register_class(StopHandCamOperator)
    bpy.utils.register_class(HandCamPanel)
    
def unregister():
    bpy.utils.unregister_class(StartHandCamOperator)
    bpy.utils.unregister_class(StopHandCamOperator)
    bpy.utils.unregister_class(HandCamPanel)
    
if __name__ == "__main__":
    register()
    