import cv2
import numpy as np
import pyautogui

# Open the webcam
cap = cv2.VideoCapture(0)

# Set the background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Apply the background subtractor to the frame
    fgmask = fgbg.apply(frame)

    # Find the contours in the foreground mask
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)

        # Get the coordinates of the center of the contour
        x, y, w, h = cv2.boundingRect(c)
        center = (x + w//2, y + h//2)

        # Get the current position of the mouse
        mouseX, mouseY = pyautogui.position()

        # Move the mouse based on the contour's position
        pyautogui.moveRel(center[0] - mouseX, center[1] - mouseY)

    # Display the frame
    cv2.imshow("Webcam", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
