import cv2

def detect_faces(image_path):
  """Detects faces in an image.

  Args:
    image_path: The path to the image file.

  Returns:
    A list of tuples, where each tuple contains the coordinates of a detected face.
  """

  face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
  img = cv2.imread(image_path)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, 1.1, 4)

  return faces

def draw_faces(image_path, faces):
  """Draws rectangles around detected faces in an image.

  Args:
    image_path: The path to the image file.
    faces: A list of tuples, where each tuple contains the coordinates of a detected face.
  """

  img = cv2.imread(image_path)
  for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
  cv2.imshow('Faces', img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

if __name__ == '__main__':
  image_path = 'path/to/your/image.jpg'
  faces = detect_faces(image_path)
  draw_faces(image_path, faces)
