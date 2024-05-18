import cv2
import os

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Function to detect faces in an image
def detect_faces(image_path):
    # Read the input image
    img = cv2.imread(image_path)
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return img

# Path to the directory containing images
dataset_dir = 'lfw_funneled/Aaron_Eckhart'

# Iterate over all images in the dataset directory
for filename in os.listdir(dataset_dir):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        # Full path to the image file
        image_path = os.path.join(dataset_dir, filename)
        # Detect faces in the image
        result_image = detect_faces(image_path)
        # Display the output
        cv2.imshow(filename, result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
