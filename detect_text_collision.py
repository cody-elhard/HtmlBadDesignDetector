from selenium import webdriver
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

def is_text_overflowing(chromedriver_binary_path, filepath, visualize_output=True):
  driver = webdriver.Chrome(chromedriver_binary_path)
  driver.get(filepath)
  driver.save_screenshot("test.png")
  image = cv2.imread("test.png")

  trained_network_input_size = 320 # x 320

  (H, W) = image.shape[:2]
  rW = W / float(trained_network_input_size)
  rH = H / float(trained_network_input_size)

  orig = image.copy()
  orig_canny = image.copy()
  image = cv2.resize(image, (trained_network_input_size, trained_network_input_size))

  net = cv2.dnn.readNet("models/frozen_east_text_detection.pb")
  blob = cv2.dnn.blobFromImage(image, 1.0, (trained_network_input_size, trained_network_input_size), (123.68, 116.78, 103.94), True, False)

  outputLayers = []
  outputLayers.append("feature_fusion/Conv_7/Sigmoid")
  outputLayers.append("feature_fusion/concat_3")

  net.setInput(blob)
  output = net.forward(outputLayers)
  scores = output[0]
  geometry = output[1]
  (numRows, numCols) = scores.shape[2:4]

  driver.close()

  rects = []
  confidences = []

  # loop over the number of rows
  for y in range(0, numRows):
      # extract the scores (probabilities), followed by the geometrical
      # data used to derive potential bounding box coordinates that surround text
      scoresData = scores[0, 0, y]
      xData0 = geometry[0, 0, y]
      xData1 = geometry[0, 1, y]
      xData2 = geometry[0, 2, y]
      xData3 = geometry[0, 3, y]
      anglesData = geometry[0, 4, y]

      # loop over the number of columns
      for x in range(0, numCols):
          # if our score does not have sufficient probability, ignore it
          if scoresData[x] < 0.5:
              continue

          # compute the offset factor as our resulting feature maps will
          # be 4x smaller than the input image
          (offsetX, offsetY) = (x * 4.0, y * 4.0)

          # extract the rotation angle for the prediction and then
          # compute the sin and cosine
          angle = anglesData[x]
          cos = np.cos(angle)
          sin = np.sin(angle)

          # use the geometry volume to derive the width and height of
          # the bounding box
          h = xData0[x] + xData2[x]
          w = xData1[x] + xData3[x]

          # compute both the starting and ending (x, y)-coordinates for
          # the text prediction bounding box
          endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
          endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
          startX = int(endX - w)
          startY = int(endY - h)

          # add the bounding box coordinates and probability score to
          # our respective lists
          rects.append((startX, startY, endX, endY))
          confidences.append(scoresData[x])

  # apply non-maxima suppression to suppress weak, overlapping bounding boxes
  boxes = non_max_suppression(np.array(rects), probs=confidences)

  text_container_boxes = []
  # loop over the bounding boxes
  for (startX, startY, endX, endY) in boxes:
      # scale the bounding box coordinates based on the respective
      # ratios
      startX = int(startX * rW)
      startY = int(startY * rH)
      endX = int(endX * rW)
      endY = int(endY * rH)

      # draw the bounding box on the image
      cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
      text_container_boxes.append([range(startX, endX), range(startY, endY)])

  # Output canny over the top
  gray = cv2.cvtColor(orig_canny, cv2.COLOR_BGR2GRAY) 
  edges = cv2.Canny(gray, 50, 150, apertureSize = 3)
  cv2.imshow("canny", edges)
  lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 5, lines=np.array([]), minLineLength=70, maxLineGap=1)

  collision = False

  if (lines.any()):
    for line in lines:
      for x1,y1,x2,y2 in line:
        cv2.line(orig,(x1,y1),(x2,y2),(255,0,0),5)

        for container in text_container_boxes:
          x_range = container[0]
          y_range = container[1]

          x_match = False
          y_match = False

          if (x1 in x_range or x2 in x_range):
            x_match = True
          if (y1 in y_range or y2 in y_range):
            y_match = True

          if (x_match and y_match):
            collision = True

  if (visualize_output):
    cv2.imshow("Collision Detection", orig)
    cv2.waitKey(0)

  return collision