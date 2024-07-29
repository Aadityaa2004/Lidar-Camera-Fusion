# config.py

# Camera Configuration
CAMERA_FOV_H = 102  # Horizontal Field of View in degrees
CAMERA_RESOLUTION_WIDTH = 4608  # Camera resolution width in pixels
CAMERA_RESOLUTION_HEIGHT = 2592  # Camera resolution height in pixels
CAMERA_FOCAL_LENGTH = 2.75  # Focal Length in mm
CAMERA_SENSOR_WIDTH = 6.3  # Sensor Width in mm (example value, adjust if needed)
CAMERA_SENSOR_HEIGHT = 3.53  # Sensor Height in mm (example value, adjust if needed)
NUM_SEGMENTS = 36 # Number of segments

# LiDAR Configuration
LIDAR_PORT = "/dev/tty.usbserial-0001"
LIDAR_BAUDRATE = 256000

# YOLO Model Path
YOLO_MODEL_PATH = '/Users/aaditya/ALSTOM/Lidar/YOLO-Weights/yolov8n.pt'

# Class Names for YOLO
CLASS_NAMES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]
