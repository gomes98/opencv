# import cv2
# import numpy as np
# from PIL import ImageGrab

# def motion_detector_gist():
# # capturing video
#     capture = cv2.VideoCapture(0)
    
#     previous_frame = None

#     while capture.isOpened():

#         # 1. Load image; convert to RGB
#         _, img_brg = capture.read()
#         img_rgb = cv2.cvtColor(src=img_brg, code=cv2.COLOR_BGR2RGB)


#         # 2. Prepare image; grayscale and blur
#         prepared_frame = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
#         prepared_frame = cv2.GaussianBlur(src=prepared_frame, ksize=(5, 5), sigmaX=0)

#         # 2. Calculate the difference
#         if (previous_frame is None):
#             # First frame; there is no previous one yet
#             previous_frame = prepared_frame
#             continue

#         # 3. calculate difference and update previous frame
#         diff_frame = cv2.absdiff(src1=previous_frame, src2=prepared_frame)
#         previous_frame = prepared_frame

#         # 4. Dilute the image a bit to make differences more seeable; more suitable for contour detection
#         kernel = np.ones((5, 5))
#         diff_frame = cv2.dilate(diff_frame, kernel, 1)

#         # 5. Only take different areas that are different enough (>20 / 255)
#         thresh_frame = cv2.threshold(src=diff_frame, thresh=20, maxval=255, type=cv2.THRESH_BINARY)[1]

#         # 6. Find and optionally draw contours
#         contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
#         # Comment below to stop drawing contours
#         cv2.drawContours(image=img_rgb, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
#         # Uncomment 6 lines below to stop drawing rectangles
#         for contour in contours:
#           area = cv2.contourArea(contour)
#           if area < 100:
#             # too small: skip!
#               continue
#         #   (x, y, w, h) = cv2.boundingRect(contour)
#         #   cv2.rectangle(img=img_rgb, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)
#           print("movimento", area)

#         cv2.imshow('Motion detector', img_rgb)

#         if (cv2.waitKey(30) == 27):
#             # out.release()
#             break

#     # Cleanup
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#   motion_detector_gist()

# 2ª versão
# ---------------------------------------------------------------------------------

# # import the opencv module
# import cv2

# # capturing video
# capture = cv2.VideoCapture(0)

# while capture.isOpened():
#     # to read frame by frame
#     _, img_1 = capture.read()
#     _, img_2 = capture.read()

#     # find difference between two frames
#     diff = cv2.absdiff(img_1, img_2)

#     # to convert the frame to grayscale
#     diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

#     # apply some blur to smoothen the frame
#     diff_blur = cv2.GaussianBlur(diff_gray, (5, 5), 0)

#     # to get the binary image
#     _, thresh_bin = cv2.threshold(diff_blur, 20, 255, cv2.THRESH_BINARY)

#     # to find contours
#     contours, hierarchy = cv2.findContours(thresh_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     # to draw the bounding box when the motion is detected
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         if cv2.contourArea(contour) > 300:
#             cv2.rectangle(img_1, (x, y), (x+w, y+h), (0, 255, 0), 2)
#     # cv2.drawContours(img_1, contours, -1, (0, 255, 0), 2)

#     # display the output
#     cv2.imshow("Detecting Motion...", img_1)
#     if cv2.waitKey(100) == 13:
#         exit()


# 3ª versão
# import cv2

# cap = cv2.VideoCapture(0)

# mog = cv2.createBackgroundSubtractorMOG2()

# while True:
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     fgmask = mog.apply(gray)
    
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     fgmask = cv2.erode(fgmask, kernel, iterations=1)
#     fgmask = cv2.dilate(fgmask, kernel, iterations=1)
    
#     contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     for contour in contours:
#         # Ignore small contours
#         if cv2.contourArea(contour) < 1000:
#             continue
        
#         # Draw bounding box around contour
#         x, y, w, h = cv2.boundingRect(contour)
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         print("movimento", cv2.contourArea(contour))
    
#     cv2.imshow('Motion Detection', frame)
#     if cv2.waitKey(1) == ord('q'):
#         break
        
# cap.release()
# cv2.destroyAllWindows()

# 4ª versão
# Python program to implement
# Webcam Motion Detector

# importing OpenCV, time and Pandas library
import cv2, time
# , pandas
# importing datetime class from datetime library
from datetime import datetime

# Assigning our static_back to None
static_back = None

# List when any moving object appear
motion_list = [ None, None ]

# Time of movement
time = []

# Initializing DataFrame, one column is start
# time and other column is end time
# df = pandas.DataFrame(columns = ["Start", "End"])

# Capturing video
video = cv2.VideoCapture(0)

# Infinite while loop to treat stack of image as video
while True:
	# Reading frame(image) from video
	check, frame = video.read()

	# Initializing motion = 0(no motion)
	motion = 0

	# Converting color image to gray_scale image
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Converting gray scale image to GaussianBlur
	# so that change can be find easily
	gray = cv2.GaussianBlur(gray, (21, 21), 0)

	# In first iteration we assign the value
	# of static_back to our first frame
	if static_back is None:
		static_back = gray
		continue

	# Difference between static background
	# and current frame(which is GaussianBlur)
	diff_frame = cv2.absdiff(static_back, gray)

	# If change in between static background and
	# current frame is greater than 30 it will show white color(255)
	thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
	thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2)

	# Finding contour of moving object
	cnts,_ = cv2.findContours(thresh_frame.copy(),
					cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	for contour in cnts:
		if cv2.contourArea(contour) < 10000:
			continue
		motion = 1

		(x, y, w, h) = cv2.boundingRect(contour)
		# making green rectangle around the moving object
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

	# Appending status of motion
	motion_list.append(motion)

	motion_list = motion_list[-2:]

	# Appending Start time of motion
	if motion_list[-1] == 1 and motion_list[-2] == 0:
		time.append(datetime.now())

	# Appending End time of motion
	if motion_list[-1] == 0 and motion_list[-2] == 1:
		time.append(datetime.now())

	# Displaying image in gray_scale
	cv2.imshow("Gray Frame", gray)

	# Displaying the difference in currentframe to
	# the staticframe(very first_frame)
	cv2.imshow("Difference Frame", diff_frame)

	# Displaying the black and white image in which if
	# intensity difference greater than 30 it will appear white
	cv2.imshow("Threshold Frame", thresh_frame)

	# Displaying color frame with contour of motion of object
	cv2.imshow("Color Frame", frame)

	key = cv2.waitKey(1)
	# if q entered whole process will stop
	if key == ord('q'):
		# if something is movingthen it append the end time of movement
		if motion == 1:
			time.append(datetime.now())
		break

# Appending time of motion in DataFrame
# for i in range(0, len(time), 2):
	# df = df.append({"Start":time[i], "End":time[i + 1]}, ignore_index = True)

# Creating a CSV file in which time of movements will be saved
# df.to_csv("Time_of_movements.csv")

video.release()

# Destroying all the windows
cv2.destroyAllWindows()
