import pandas as pd
import numpy as np
import os
import cv2


def bilateralFilter(image, d, sig):
    image = cv2.bilateralFilter(image, d, sig, sig)
    return image

def threshold(image, t):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, image = cv2.threshold(image, t, 255, cv2.THRESH_BINARY_INV)
#     ret, image = cv2.threshold(image, t, 255, cv2.THRESH_BINARY)
    return image

def dilate(image, kernalSize):
    kernel = np.ones(kernalSize, np.uint8)
    kernel
    image = cv2.dilate(image, kernel, iterations=1)
    return image

ANCHOR_POINT = 6000
MIDZONE_THRESHOLD = 15000
MIN_HANDWRITING_HEIGHT_PIXEL = 20

def horizontalProjection(img):
    # Return a list containing the sum of the pixels in each row
    (h, w) = img.shape[:2]
    sumRows = []
    for j in range(h):
        row = img[j:j+1, 0:w]  # y1:y2, x1:x2
        sumRows.append(np.sum(row))
    return sumRows


''' function to calculate vertical projection of the image pixel columns and return it '''


def verticalProjection(img):
    # Return a list containing the sum of the pixels in each column
    (h, w) = img.shape[:2]
    sumCols = []
    for j in range(w):
        col = img[0:h, j:j+1]  # y1:y2, x1:x2
        sumCols.append(np.sum(col))
    return sumCols


''' function to extract lines of handwritten text from the image using horizontal projection '''


def extractLines(img):

    global LETTER_SIZE
    global LINE_SPACING
    global TOP_MARGIN
    # apply bilateral filter
    dim = img.shape
    #print(dim)
    filtered = bilateralFilter(img, 5,50)

    # convert to grayscale and binarize the image by INVERTED binary thresholding
    # it's better to clear unwanted dark areas at the document left edge and use a high threshold value to preserve more text pixels
    thresh = threshold(filtered, 160)
    # cv2.imshow('thresh', lthresh)

    # extract a python list containing values of the horizontal projection of the image into 'hp'
    hpList = horizontalProjection(thresh)

    # Extracting 'Top Margin' feature.
    topMarginCount = 0
    for sum in hpList:
        # sum can be strictly 0 as well. Anyway we take 0 and 255.
        if (sum <= 255):
            topMarginCount += 1
        else:
            break

    # print "(Top margin row count: "+str(topMarginCount)+")"

    # FIRST we extract the straightened contours from the image by looking at occurance of 0's in the horizontal projection.
    lineTop = 0
    lineBottom = 0
    spaceTop = 0
    spaceBottom = 0
    indexCount = 0
    setLineTop = True
    setSpaceTop = True
    includeNextSpace = True
    space_zero = []  # stores the amount of space between lines
    lines = []  # a 2D list storing the vertical start index and end index of each contour

    # we are scanning the whole horizontal projection now
    for i, sum in enumerate(hpList):
        # sum being 0 means blank space
        if (sum == 0):
            if (setSpaceTop):
                spaceTop = indexCount
                setSpaceTop = False  # spaceTop will be set once for each start of a space between lines
            indexCount += 1
            spaceBottom = indexCount
            if (i < len(hpList)-1):  # this condition is necessary to avoid array index out of bound error
                # if the next horizontal projectin is 0, keep on counting, it's still in blank space
                if (hpList[i+1] == 0):
                    continue
            # we are using this condition if the previous contour is very thin and possibly not a line
            if (includeNextSpace):
                space_zero.append(spaceBottom-spaceTop)
            else:
                if (len(space_zero) == 0):
                    previous = 0
                else:
                    previous = space_zero.pop()
                space_zero.append(previous + spaceBottom-lineTop)
            # next time we encounter 0, it's begining of another space so we set new spaceTop
            setSpaceTop = True

        # sum greater than 0 means contour
        if (sum > 0):
            if (setLineTop):
                lineTop = indexCount
                setLineTop = False  # lineTop will be set once for each start of a new line/contour
            indexCount += 1
            lineBottom = indexCount
            if (i < len(hpList)-1):  # this condition is necessary to avoid array index out of bound error
                # if the next horizontal projectin is > 0, keep on counting, it's still in contour
                if (hpList[i+1] > 0):
                    continue

                # if the line/contour is too thin <10 pixels (arbitrary) in height, we ignore it.
                # Also, we add the space following this and this contour itself to the previous space to form a bigger space: spaceBottom-lineTop.
                if (lineBottom-lineTop < 20):
                    includeNextSpace = False
                    setLineTop = True  # next time we encounter value > 0, it's begining of another line/contour so we set new lineTop
                    continue
            # the line/contour is accepted, new space following it will be accepted
            includeNextSpace = True

            # append the top and bottom horizontal indices of the line/contour in 'lines'
            lines.append([lineTop, lineBottom])
            setLineTop = True  # next time we encounter value > 0, it's begining of another line/contour so we set new lineTop

    '''
	# Printing the values we found so far.
	for i, line in enumerate(lines):
		print
		print i
		print line[0]
		print line[1]
		print len(hpList[line[0]:line[1]])
		print hpList[line[0]:line[1]]
	
	for i, line in enumerate(lines):
		cv2.imshow("line "+str(i), img[line[0]:line[1], : ])
	'''

    # SECOND we extract the very individual lines from the lines/contours we extracted above.
    fineLines = []  # a 2D list storing the horizontal start index and end index of each individual line
    for i, line in enumerate(lines):

        # 'anchor' will locate the horizontal indices where horizontal projection is > ANCHOR_POINT for uphill or < ANCHOR_POINT for downhill(ANCHOR_POINT is arbitrary yet suitable!)
        anchor = line[0]
        anchorPoints = []  # python list where the indices obtained by 'anchor' will be stored
        # it implies that we expect to find the start of an individual line (vertically), climbing up the histogram
        upHill = True
        # it implies that we expect to find the end of an individual line (vertically), climbing down the histogram
        downHill = False
        # we put the region of interest of the horizontal projection of each contour here
        segment = hpList[line[0]:line[1]]

        for j, sum in enumerate(segment):
            if (upHill):
                if (sum < ANCHOR_POINT):
                    anchor += 1
                    continue
                anchorPoints.append(anchor)
                upHill = False
                downHill = True
            if (downHill):
                if (sum > ANCHOR_POINT):
                    anchor += 1
                    continue
                anchorPoints.append(anchor)
                downHill = False
                upHill = True

        # print anchorPoints

        # we can ignore the contour here
        if (len(anchorPoints) < 2):
            continue

        '''
		# the contour turns out to be an individual line
		if(len(anchorPoints)<=3):
			fineLines.append(line)
			continue
		'''
        # len(anchorPoints) > 3 meaning contour composed of multiple lines
        lineTop = line[0]
        for x in range(1, len(anchorPoints)-1, 2):
            # 'lineMid' is the horizontal index where the segmentation will be done
            lineMid = (anchorPoints[x]+anchorPoints[x+1])/2
            lineBottom = lineMid
            # line having height of pixels <20 is considered defects, so we just ignore it
            # this is a weakness of the algorithm to extract lines (anchor value is ANCHOR_POINT, see for different values!)
            if (lineBottom-lineTop < 20):
                continue
            fineLines.append([lineTop, lineBottom])
            lineTop = lineBottom
        if (line[1]-lineTop < 20):
            continue
        fineLines.append([lineTop, line[1]])

    # LINE SPACING and LETTER SIZE will be extracted here
    # We will count the total number of pixel rows containing upper and lower zones of the lines and add the space_zero/runs of 0's(excluding first and last of the list ) to it.
    # We will count the total number of pixel rows containing midzones of the lines for letter size.
    # For this, we set an arbitrary (yet suitable!) threshold MIDZONE_THRESHOLD = 15000 in horizontal projection to identify the midzone containing rows.
    # These two total numbers will be divided by number of lines (having at least one row>MIDZONE_THRESHOLD) to find average line spacing and average letter size.
    space_nonzero_row_count = 0
    midzone_row_count = 0
    lines_having_midzone_count = 0
    flag = False
    #print(hpList)
    #print(len(hpList))
    #print((fineLines))
    #print(line[1])
    fineLines1 = [[int(value) for value in sublist] for sublist in fineLines]

    #print(hpList[line[0]:line[1]-5])
    for i, line in enumerate(fineLines1):
        segment = hpList[line[0]:line[1]]
        for j, sum in enumerate(segment):
            if (sum < MIDZONE_THRESHOLD):
                space_nonzero_row_count += 1
            else:
                midzone_row_count += 1
                flag = True

        # This line has contributed at least one count of pixel row of midzone
        if (flag):
            lines_having_midzone_count += 1
            flag = False

    # error prevention ^-^
    if (lines_having_midzone_count == 0):
        lines_having_midzone_count = 1

    # excluding first and last entries: Top and Bottom margins
    total_space_row_count = space_nonzero_row_count + np.sum(space_zero[1:-1])
    # the number of spaces is 1 less than number of lines but total_space_row_count contains the top and bottom spaces of the line
    average_line_spacing = float(
        total_space_row_count) / lines_having_midzone_count
    average_letter_size = float(midzone_row_count) / lines_having_midzone_count
    # letter size is actually height of the letter and we are not considering width
    LETTER_SIZE = average_letter_size
    # error prevention ^-^
    if (average_letter_size == 0):
        average_letter_size = 1
    # We can't just take the average_line_spacing as a feature directly. We must take the average_line_spacing relative to average_letter_size.
    # Let's take the ratio of average_line_spacing to average_letter_size as the LINE SPACING, which is perspective to average_letter_size.
    relative_line_spacing = average_line_spacing / average_letter_size
    LINE_SPACING = relative_line_spacing

    # Top marging is also taken relative to average letter size of the handwritting
    relative_top_margin = float(topMarginCount) / average_letter_size
    TOP_MARGIN = relative_top_margin

    '''
	# showing the final extracted lines
	for i, line in enumerate(fineLines):
		cv2.imshow("line "+str(i), img[line[0]:line[1], : ])
	'''

    # print space_zero
    # print lines
    # print fineLines
    # print midzone_row_count
    # print total_space_row_count
    # print len(hpList)
    # print average_line_spacing
    # print lines_having_midzone_count
    # print i
    '''
	print ("Average letter size: "+str(average_letter_size))
	print ("Top margin relative to average letter size: "+str(relative_top_margin))
	print ("Average line spacing relative to average letter size: "+str(relative_line_spacing))
	'''
    return LETTER_SIZE


def letter_size_extract(file_name):
    image = cv2.imread(file_name)
    letter_size = extractLines(image)
	
    return [letter_size] 