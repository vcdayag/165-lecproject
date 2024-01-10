import cv2 

def detectShape(img: cv2.typing.MatLike, mask: cv2.typing.MatLike):
    # reading image 
    # img = cv2.imread('data/IMG_20231220_132420.jpg') 

    # converting image into grayscale image 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    countourmask = mask.copy()

    # setting threshold of gray image 
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) 

    # using a findContours() function 
    contours, _ = cv2.findContours( 
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

    # here we are ignoring first counter because 
    # findcontour function detects whole image as shape 
    contours = contours[1:]
    
    AREA_MIN = 2000
    AREA_MAX = 5000

    # list for storing names of shapes 
    for contour in contours:
        _,_,w,h = cv2.boundingRect(contour)
        contourarea = w*h
        
        if contourarea < AREA_MIN or contourarea > AREA_MAX:
            continue

        # cv2.approxPloyDP() function to approximate the shape 
        approx = cv2.approxPolyDP( 
            contour, 0.01 * cv2.arcLength(contour, True), True) 
        
        # using drawContours() function 
        # cv2.drawContours(img, [contour], 0, (0, 0, 255), 5) 
        cv2.drawContours(countourmask, [contour], -1, (0,0,0), -1)

        # finding center point of shape 
        M = cv2.moments(contour) 
        x = 0
        y = 0
        if M['m00'] != 0.0: 
            x = int(M['m10']/M['m00']) 
            y = int(M['m01']/M['m00']) 

        # putting shape name at center of each shape 
        if len(approx) == 4: 
            cv2.putText(img, 'Quadrilateral', (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) 
        # if len(approx) == 3: 
        #     cv2.putText(img, 'Triangle', (x, y), 
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) 

        # elif len(approx) == 4: 
        #     cv2.putText(img, 'Quadrilateral', (x, y), 
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) 

        # elif len(approx) == 5: 
        #     cv2.putText(img, 'Pentagon', (x, y), 
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) 

        # elif len(approx) == 6: 
        #     cv2.putText(img, 'Hexagon', (x, y), 
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) 

        # else: 
        #     cv2.putText(img, 'circle', (x, y), 
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) 

    # displaying the image after drawing contours 
    finalmask = cv2.bitwise_xor(countourmask,mask)
    
    finalimage = cv2.bitwise_and(img, img, mask=finalmask)
    cv2.imshow('shapes', finalimage) 

    cv2.waitKey(0) 
    cv2.destroyAllWindows() 

if __name__ == "__main__":
    img = cv2.imread('data/IMG_20231220_132420.jpg')
    detectShape(img)