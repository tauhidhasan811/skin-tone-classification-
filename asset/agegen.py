import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt

def get_age_gen(image_path):
    result = DeepFace.analyze(img_path=image_path, actions=['age', 'gender'], enforce_detection=False)
    print("Analysis Result:\n", result)

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if isinstance(result, list):
        gender = result[0]['dominant_gender']
        age = result[0]['age']
    else:
        gender = result['dominant_gender']
        age = result['age']

    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Predicted: {gender}, Age: {age}")
    plt.show()