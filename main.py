# import the opencv library
import cv2
import dlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def eye_extract(roi, landmarks, points):
    region = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in points])
    region = region.astype(np.int32)

    min_x = np.min(region[:, 0])
    max_x = np.max(region[:, 0])
    min_y = np.min(region[:, 1])
    max_y = np.max(region[:, 1])

    while max_x + 1 - min_x < 40:
        max_x += 1
        min_x -= 1
    if max_x + 1 - min_x > 40:
        max_x -= 1
    while max_y + 1 - min_y < 20:
        max_y += 1
        min_y -= 1
    if max_y + 1 - min_y > 20:
        max_y -= 1

    if max_x + 1 - min_x != 40 or max_y + 1 - min_y != 20:
        return None, None

    roi = frame[min_y:max_y + 1, min_x:max_x + 1]
    return roi, (min_x, min_y, max_x, max_y)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(3200, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

model = Net()
model.load_state_dict(torch.load('pupil_center_model_cpu.pth'))
model.eval()

face_detector = dlib.get_frontal_face_detector()
shape_model = "BANDIT_Eye_data_code/trained_models/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(shape_model)


LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]
FACE_BOX_IMG_H = 20
FACE_BOX_IMG_W = 40

# define a video capture object
vid = cv2.VideoCapture(0)

while (True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    faces = face_detector(gray)
    if len(faces) == 0:
        continue

    face0 = faces[0]
    face_size = (face0.right() - face0.left()) * 0.2
    face_img = frame[face0.top():face0.bottom(), face0.left():face0.right()]
    landmarks = predictor(frame, face0)
    #
    left_eye_img, left_eye_rect = eye_extract(frame, landmarks, LEFT_EYE_POINTS)
    right_eye_img, right_eye_rect = eye_extract(frame, landmarks, RIGHT_EYE_POINTS)

    if left_eye_img is None or right_eye_img is None:
        continue


    test_input = np.stack([left_eye_img, right_eye_img])
    test_input = test_input.astype(np.float32)/255.
    # Shape: [2, 3, 20, 40]
    test_input = torch.from_numpy(test_input).permute([0, 3, 1, 2])
    with torch.no_grad():
        preds = model(test_input)
    preds = (preds.numpy() * max([FACE_BOX_IMG_H, FACE_BOX_IMG_W])).astype(int)

    cv2.circle(frame, (preds[0, 0]+left_eye_rect[0], preds[0, 1]+left_eye_rect[1]), radius=1, color=(0, 254, 0), thickness=-1)
    cv2.circle(frame, (preds[1, 0]+right_eye_rect[0], preds[1, 1]+right_eye_rect[1]), radius=1, color=(0, 254, 0), thickness=-1)

    cv2.circle(left_eye_img, (preds[0, 0], preds[0, 1]), radius=1, color=(0, 254, 0), thickness=-1)
    cv2.circle(right_eye_img, (preds[1, 0], preds[1, 1]), radius=1, color=(0, 254, 0), thickness=-1)

    left_eye_img = cv2.resize(left_eye_img, [left_eye_img.shape[1]*5, left_eye_img.shape[0]*5])
    right_eye_img = cv2.resize(right_eye_img, [right_eye_img.shape[1]*5, right_eye_img.shape[0]*5])


    # canvas = np.zeros_like(frame)
    margin = int(frame.shape[0] * 0.1)
    canvas = np.zeros([frame.shape[0] + margin*2 + left_eye_img.shape[0], frame.shape[1], 3], dtype=frame.dtype) + 255
    canvas[:frame.shape[0], :, :] = frame
    canvas_center = int(canvas.shape[1] * 0.5)
    canvas[frame.shape[0] + margin: frame.shape[0] + margin + left_eye_img.shape[0], canvas_center-margin-left_eye_img.shape[1]:canvas_center-margin] = left_eye_img
    canvas[frame.shape[0] + margin: frame.shape[0] + margin + right_eye_img.shape[0], canvas_center+margin: canvas_center+margin+right_eye_img.shape[1]] = right_eye_img

    # canvas2 = np.zeros([canvas.shape[0] + 20, canvas.shape[1]+20, 3], dtype=canvas.dtype) + 255
    # canvas2[10:-10, 10:-10] = canvas
    #
    # Display the resulting frame
    # cv2.imshow('frame', frame)
    cv2.imshow('frame', canvas)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()