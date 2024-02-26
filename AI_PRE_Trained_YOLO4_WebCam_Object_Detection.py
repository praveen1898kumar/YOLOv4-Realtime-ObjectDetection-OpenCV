import cv2  # Import the OpenCV library
import numpy as np  # Import the NumPy library for numerical operations

# Load YOLO (You Only Look Once) model
net = cv2.dnn.readNet("/Users/praveen18kumar/Downloads/yolov4.weights", "/Users/praveen18kumar/Downloads/yolov4.cfg")

# Load the classes that YOLO can detect from a file
classes = []
with open("/Users/praveen18kumar/Downloads/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get the names of all the layers in the network   
layer_names = net.getLayerNames()

# Get the names of the output layers of the network
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialize video capture from webcam
cap = cv2.VideoCapture(0)  # Change the argument to the appropriate video source if not webcam

while True: # Infinite loop to continuously process frames
    # Initialize list to store indices of objects that have been drawn
    drawn_objects = []

    # Capture frame-by-frame from the video source
    ret, frame = cap.read()
    if not ret: # Break the loop if frame capture fails
        break

    # Resize frame for faster processing (optional)
    frame = cv2.resize(frame, None, fx=0.4, fy=0.4)
    height, width, channels = frame.shape

   # Detecting objects in the frame
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing information on the screen
    for out in outs:
        for detection in out:
            scores = detection[5:] # Confidence scores for each class
            class_id = np.argmax(scores) # Get the index of the class with the highest score
            confidence = scores[class_id] # Confidence score for the detected class
            if confidence > 0.5: # Threshold for considering a detection valid
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Check if this object has already been drawn
                if class_id not in drawn_objects:
                    # Draw rectangle and text around the detected object
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    cv2.putText(frame, classes[class_id], (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1)

                    # Add index of drawn object to the list
                    drawn_objects.append(class_id)

    # Display the resulting frame with detected objects
    cv2.imshow('Object Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
