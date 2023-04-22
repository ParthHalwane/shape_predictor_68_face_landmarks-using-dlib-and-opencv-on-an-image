import dlib
import cv2

# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load the image
image = cv2.imread("sample_pic.JPG")

# Resize the image
image = cv2.resize(image, (0, 0), fx=0.4, fy=0.4)  # Change the values of fx and fy to resize the image as desired

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = detector(gray)

# Loop through each face
for face in faces:
    # Get the landmarks/parts for the face in box d.
    landmarks = predictor(gray, face)

    # Loop through each landmark and draw a circle
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

# Show the image
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
