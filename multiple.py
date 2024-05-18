import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Function to detect faces in an image
def detect_faces(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return image

# Function to process a single image
def process_single_image(image_path):
    # Read the input image
    img = cv2.imread(image_path)

    result_image = detect_faces(img)

    cv2.imshow("Detected Faces", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Path to the image file
image_path = 'lfw_funneled/Bill_Rainer_NEW/always_sunny.jpg'

# Process the single image
process_single_image(image_path)
