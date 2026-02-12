import cv2
import os

# Path to the Haar Cascade XML file
HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Load the Haar Cascade
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

# Function to crop faces from an image
def crop_faces(image_path, output_dir, undetected_file):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return

    # Convert the image to grayscale (required by Haar Cascade)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If no faces are detected, print and log the file name
    if len(faces) == 0:
        print(f"No face detected in: {image_path}")
        with open(undetected_file, "a") as f:
            f.write(f"{image_path}\n")
        return

    # Loop through the detected faces
    for i, (x, y, w, h) in enumerate(faces):
        # Crop the face region
        face = img[y:y + h, x:x + w]

        # Save the cropped face image
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(output_dir, f"{name}{ext}")
        cv2.imwrite(output_path, face)
        print(f"Face saved: {output_path}")

# Directory containing input images
input_dir = "../dataset/lfw-1/male"

# Directory to save cropped face images
output_dir = "../dataset/lfw-2/male"
os.makedirs(output_dir, exist_ok=True)

# Path to the file logging undetected faces
undetected_file = os.path.join(output_dir, "undetected_faces.txt")
if os.path.exists(undetected_file):
    os.remove(undetected_file)

# Process each image in the input directory
for file_name in os.listdir(input_dir):
    file_path = os.path.join(input_dir, file_name)

    # Skip non-image files
    if not file_name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    crop_faces(file_path, output_dir, undetected_file)

# Path to the .txt file with paths to images to be removed
to_remove_file = "undetected_faces_male.txt"

# Read the file and remove the listed images
if os.path.exists(to_remove_file):
    with open(to_remove_file, "r") as f:
        paths_to_remove = f.readlines()
        
    for path in paths_to_remove:
        path = path.strip()  # Remove any leading/trailing whitespace or newline characters
        if os.path.exists(path):
            os.remove(path)
            print(f"Removed file: {path}")
        else:
            print(f"File not found, could not remove: {path}")
else:
    print(f"No file found at {to_remove_file}, nothing to remove.")