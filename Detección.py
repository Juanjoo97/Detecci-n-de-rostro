import cv2
import numpy as np
# Objeto de video
captura = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

while True:
        # Capturamos el frame
        (grabbed, image) = captura.read()
        #salir video
        if not grabbed:
            break

        #Escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
        idx=0
        for (x,y,w,h) in faces:
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
            idx = idx +1
            cv2.putText(image, "Face #" + str(idx),(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),1)

        #Mostrar imagen
        cv2.imshow('Video',image)
        tecla = cv2.waitKey(25)& 0xFF
        # Salimos si la tecla presionada es ESC
        if tecla == 27:
                 break
# Liberamos objeto
captura.release()
# Destruimos ventanas
cv2.destroyAllWindows()