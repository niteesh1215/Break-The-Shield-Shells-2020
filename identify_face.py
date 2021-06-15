import face_recognition
from PIL import Image,ImageDraw,ImageFont
from face_recognition.api import face_locations

image_of_niteesh = face_recognition.load_image_file('./known/Niteesh Mahato.jpg')
niteesh_face_encoding = face_recognition.face_encodings(image_of_niteesh)[0]

image_of_blins = face_recognition.load_image_file('./known/blins.jpeg')
blins_face_encoding = face_recognition.face_encodings(image_of_blins)[0]

image_of_komal = face_recognition.load_image_file('./known/komal.jpeg')
komal_face_encoding = face_recognition.face_encodings(image_of_komal)[0]

image_of_sachin = face_recognition.load_image_file('./known/sachin.jpeg')
sachin_face_encoding = face_recognition.face_encodings(image_of_sachin)[0]


known_face_encodings = [niteesh_face_encoding, blins_face_encoding, komal_face_encoding, sachin_face_encoding]
known_face_names = ["Niteesh Mahato", "Blinsia", "Komal", "Sachin"]

# Load test image to find faces in

unknown_image = face_recognition.load_image_file('./unknown/group1.jpeg')

#unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]

#find faces in test image 

face_locations = face_recognition.face_locations(unknown_image)

face_encodings = face_recognition.face_encodings(unknown_image,face_locations)

# Convert to PIL format 
pil_image = Image.fromarray(unknown_image)

#Create  a ImageDraw instance 
draw = ImageDraw.Draw(pil_image)

font = ImageFont.load_default()
#font = ImageFont.truetype(font,20)

#ImageFont.FreeTypeFont.font_variant

#loop through faces in test image 
for(top,right,bottom,left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance= 0.42)
    name = "Unknown Person"


    # If match 
    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]

    # Draw Box 
    draw.rectangle(((left,top),(right, bottom)),outline = (0,0,0))

    # Draw label 
    text_width, text_height = font.getsize(name)

    draw.rectangle(((left,bottom - text_height -10),(right, bottom)), fill = (0,0,0), outline = (0,0,0))

    draw.text((left + 6, bottom - text_height -5), name, fill=(255,255,255))

del draw

pil_image.show()


