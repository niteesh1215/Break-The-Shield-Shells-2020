import face_recognition

image_of_niteesh = face_recognition.load_image_file('./known/Niteesh Mahato.jpg')

unknown_image = face_recognition.load_image_file('./unknown/uk.jpeg')

niteesh_face_encoding = face_recognition.face_encodings(image_of_niteesh)[0]

unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]

results = face_recognition.compare_faces([niteesh_face_encoding],unknown_face_encoding)

if results[0]:
    print('This is Niteesh Mahato')
else: 
    print('This is not Niteesh Mahato')