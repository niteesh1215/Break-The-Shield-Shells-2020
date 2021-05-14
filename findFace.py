import face_recognition

from face_recognition.api import face_locations

image = face_recognition.load_image_file('./group_photo/group-photo.png')

face_locations = face_recognition.face_locations(image)

print(f'There are {len(face_locations)} people in this image')


