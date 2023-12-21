import tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import Image, ImageTk, ImageOps
import cv2
import numpy as np
import tensorflow
from gtts import gTTS
import playsound
import wikipedia
from webdriver_manager.chrome import ChromeDriverManager
import os

wikipedia.set_lang('vi')
language = 'vi'
path = ChromeDriverManager().install()

# Initialize the camera
cam = cv2.VideoCapture(0)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

def speak(text):
    tts = gTTS(text=text, lang=language, slow=False)
    tts.save("sound.mp3")
    playsound.playsound("sound.mp3", True)
    os.remove("sound.mp3")

class FaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detection App")

        # Make the window fullscreen
        self.root.attributes('-fullscreen', True)

        # Create a button for image capture and face detection
        self.capture_button = ttk.Button(root, text="Capture and Detect Face", command=self.capture_and_detect)
        self.capture_button.pack(pady=10)

        # Create a label to display the captured image
        self.image_label = ttk.Label(root)
        self.image_label.pack()

        # Create a scrolled text widget for displaying results
        self.result_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=40, height=10)
        self.result_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Bind the Escape key to exit fullscreen
        self.root.bind("<Escape>", self.exit_fullscreen)

    def capture_and_detect(self):
        # Capture an image
        ret, frame = cam.read()
        cv2.imwrite('img_detect.png', frame)

        # Perform face detection
        result = self.face_detection()

        # Display the result
        self.display_result(result)

    def face_detection(self):
        # Load the image into the array
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image = Image.open('img_detect.png')
        image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)  # Resize using ImageOps
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array

        prediction = model.predict(data)
        print(prediction)

        name = ["Đội mũ bảo hiểm", "Không đội mũ bảo hiểm", "Không có người"]
        index = -1
        max_value = -1
        for i in range(0, len(prediction[0])):
            if max_value < prediction[0][i]:
                max_value = prediction[0][i]
                index = i

        return name[index], max_value

    def display_result(self, result):
        # Display the result on the UI
        result_text = f"Result: {result[0]}\nAccuracy: {result[1]}"
        self.result_text.insert(tk.END, result_text + "\n\n")

        # Display the captured image
        image = Image.open('img_detect.png')
        image = ImageTk.PhotoImage(image)
        self.image_label.config(image=image)
        self.image_label.image = image

        # Speak the result
        speak(result[0])

    def exit_fullscreen(self, event):
        # Exit fullscreen when Escape key is pressed
        self.root.attributes('-fullscreen', False)

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceDetectionApp(root)
    root.mainloop()
