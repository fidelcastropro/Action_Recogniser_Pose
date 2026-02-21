import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp

########################################
# CONFIGURATION
########################################
MODEL_PATH = "action_lstm1.pth"
SEQUENCE_LENGTH = 30
FEATURE_SIZE = 1662
ACTIONS = np.array([ "punch", "kick","iloveyou"])
CONFIDENCE_THRESHOLD = 0.80

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################
# LSTM MODEL DEFINITION
########################################
class ActionLSTM(nn.Module):
    def __init__(self, num_classes):
        super(ActionLSTM, self).__init__()

        self.lstm1 = nn.LSTM(FEATURE_SIZE, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 128, batch_first=True)
        self.lstm3 = nn.LSTM(128, 64, batch_first=True)

        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)

        x = x[:, -1, :]   # last timestep

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

########################################
# LOAD MODEL
########################################
model = ActionLSTM(num_classes=len(ACTIONS)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("✅ Model loaded successfully")

########################################
# MEDIAPIPE HOLISTIC SETUP
########################################
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

########################################
# LANDMARK EXTRACTION (1662 FEATURES)
########################################
def extract_keypoints(results):
    pose = np.array([[p.x, p.y, p.z, p.visibility]
                     for p in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(33 * 4)

    face = np.array([[f.x, f.y, f.z]
                     for f in results.face_landmarks.landmark]).flatten() \
        if results.face_landmarks else np.zeros(468 * 3)

    lh = np.array([[l.x, l.y, l.z]
                   for l in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21 * 3)

    rh = np.array([[r.x, r.y, r.z]
                   for r in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 3)

    return np.concatenate([pose, face, lh, rh])

########################################
# REAL-TIME WEBCAM INFERENCE
########################################
sequence = []

cap = cv2.VideoCapture(0)

print("🎥 Webcam started — Press 'q' to quit")

with holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)

        if len(sequence) > SEQUENCE_LENGTH:
            sequence.pop(0)

        if len(sequence) == SEQUENCE_LENGTH:
            input_tensor = torch.tensor(sequence, dtype=torch.float32)
            input_tensor = input_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

            predicted_action = ACTIONS[np.argmax(probs)]
            confidence = np.max(probs)

            if confidence > CONFIDENCE_THRESHOLD:
                cv2.putText(
                    image,
                    f"{predicted_action} ({confidence:.2f})",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

        cv2.imshow("Action Recognition (Live)", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("🛑 Webcam stopped")
