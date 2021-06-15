from face_recognition.api import face_distance, face_encodings, face_locations
import numpy
import face_recognition
import cv2

video_capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)

image_of_niteesh = face_recognition.load_image_file('./known/Niteesh Mahato.jpg')
niteesh_face_encoding = face_recognition.face_encodings(image_of_niteesh)[0]

image_of_blins = face_recognition.load_image_file('./known/blins.jpeg')
blins_face_encoding = face_recognition.face_encodings(image_of_blins)[0]

image_of_komal = face_recognition.load_image_file('./known/komal.jpeg')
komal_face_encoding = face_recognition.face_encodings(image_of_komal)[0]

image_of_sachin = face_recognition.load_image_file('./known/sachin.jpeg')
sachin_face_encoding = face_recognition.face_encodings(image_of_sachin)[0]

image_of_shawn = face_recognition.load_image_file('./known/shawn.jpg')
shawn_face_encoding = face_recognition.face_encodings(image_of_shawn)[0]

image_of_vijay = face_recognition.load_image_file('./known/vijay.jpeg')
vijay_face_encoding = face_recognition.face_encodings(image_of_vijay)[0]

#image_of_gitesh= face_recognition.load_image_file('./known/gitesh.jpeg')
#gitesh_face_encoding = face_recognition.face_encodings(image_of_gitesh)[0]

#image_of_jibin= face_recognition.load_image_file('./known/jibin.jpeg')
#jibin_face_encoding = face_recognition.face_encodings(image_of_jibin)[0]

image_of_jagat= face_recognition.load_image_file('./known/jagat.jpeg')
jagat_face_encoding = face_recognition.face_encodings(image_of_jagat)[0]

#image_of_romy= face_recognition.load_image_file('./known/romy.jpeg')
#romy_face_encoding = face_recognition.face_encodings(image_of_romy)[0]

known_face_encodings = [niteesh_face_encoding,shawn_face_encoding, blins_face_encoding, komal_face_encoding,vijay_face_encoding,jagat_face_encoding]
known_face_names = ["Niteesh", "Painter","Blinsia", "Komal", "vijay", "Jagat"]


while True:
    ret, frame = video_capture.read()
    rgb_frame = frame[:,:,::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame,face_locations)

    for(top,right,bottom,left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distance = face_recognition.face_distance(known_face_encodings,face_encoding)

        best_match_index = numpy.argmin(face_distance)

        if matches[best_match_index] :
            name = known_face_names[best_match_index] 

        cv2.rectangle(frame,(left,top),(right,bottom),(0,0,0),2)
        cv2.rectangle(frame,(left,bottom - 35),(right,bottom),(0,0,0),cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(frame,name,(left + 6, bottom -6),font, 0.5,(255,255,255),1)

    cv2.imshow('Webcam_face_recognition',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

video_capture.release()
cv2.destroyAllWindows()