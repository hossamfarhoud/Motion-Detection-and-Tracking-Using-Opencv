import cv2
import numpy as np

# Open the video file
cap = cv2.VideoCapture('vtest.avi')

# Get the frame width and height of the video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object to save the output
fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')  # Codec for AVI files
out = cv2.VideoWriter("output.avi", fourcc, 5.0, (1280, 720))  # Save at 5 FPS with resolution 1280x720

# Read the first two frames of the video
ret, frame1 = cap.read()  # First frame
ret, frame2 = cap.read()  # Second frame

# Print the shape of the first frame (dimensions of the frame)
print(frame1.shape)

# Loop to process video frames
while cap.isOpened():
    # Compute the absolute difference between the two frames to detect changes
    diff = cv2.absdiff(frame1, frame2)
    
    # Convert the difference image to grayscale
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise and smoothen the image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply binary thresholding to create a binary image
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    
    # Dilate the threshold image to fill gaps in detected contours
    dilated = cv2.dilate(thresh, None, iterations=3)
    
    # Find contours from the dilated image
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through each detected contour
    for contour in contours:
        # Get the bounding rectangle for the contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Ignore small contours to avoid noise (area < 900 is ignored)
        if cv2.contourArea(contour) < 900:
            continue
        
        # Draw a rectangle around the detected movement
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Add a text label to indicate movement status
        cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        #cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)
        
    # Resize the frame for saving/output
    image = cv2.resize(frame1, (1280, 720))
    
    # Write the processed frame to the output video file
    out.write(image)
    
    # Display the frame with the drawn rectangles
    cv2.imshow("feed", frame1)

    # Update frames for the next iteration
    frame1 = frame2
    ret, frame2 = cap.read()  # Read the next frame

    # Break the loop if the ESC key (key code 27) is pressed
    if cv2.waitKey(40) == 27:
        break

# Release all resources
cv2.destroyAllWindows()  # Close all OpenCV windows
cap.release()  # Release the video capture object
out.release()  # Release the video writer object


