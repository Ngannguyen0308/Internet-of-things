import tensorflow.keras 
from PIL import Image, ImageOps
import numpy as np
import cv2
import os
import playsound
import wikipedia
from gtts import gTTS
from webdriver_manager.chrome import ChromeDriverManager

wikipedia.set_lang('vi')
language = 'vi'
path = ChromeDriverManager().install()

def speakObjectName(text):
    tts = gTTS(text = text, lang=language, slow=False)
    tts.save('sound.mp3')
    playsound.playsound('sound.mp3', True)
    os.remove('sound.mp3')


cam = cv2.VideoCapture(0)
def capture_img():
    ret, frame = cam.read()
    cv2.imwrite('img_detect.png', frame)

def obj_detection():
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = tensorflow.keras.models.load_model('keras_model.h5')
            
    # Create the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open('img_detect.png')

    # resize the image to a 224x224
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # display the resized image
    image.show()

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    prediction = model.predict(data)
    print(prediction)

    name = ["Kẹp tóc", "airpod", "nothing"]
    index = -1
    max_value = -1
    for i in range(0, len(prediction[0])):
        if max_value < prediction[0][i]:
            max_value = prediction[0][1]
            index = i
    print("ket qua: ", name[index])
    print("chinh xac: ", max_value)
    speakObjectName(name[index])

while True:
    capture_img()
    obj_detection()