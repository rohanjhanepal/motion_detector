import cv2

first_frame = None

vid = cv2.VideoCapture(0)

while True:

    rect , frame = vid.read()

    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21),0)
    

    if first_frame is None:
        first_frame = gray
        continue
    

    delta_frame = cv2.absdiff(first_frame, gray)
    
    thres_frame = cv2.threshold(delta_frame , 30 ,255,cv2.THRESH_BINARY)[1]
    
    thres_frame = cv2.dilate(thres_frame , None , iterations = 3)
    edged = cv2.Canny(thres_frame , 30 , 200)

    cnts,bla = cv2.findContours(thres_frame.copy(), cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts :
        if (cv2.contourArea(contour) <1000):
            continue
        (x,y,w,h)= cv2.boundingRect(contour)
             
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

                                 
    
                                  
                                  
    
    cv2.imshow('frame' , frame)
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

vid.release()
cv2.destroyAllWindows()
