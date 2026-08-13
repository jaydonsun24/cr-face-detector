import cv2
import torch
from PIL import Image
from model import model
from data_setup import test_transform, device, class_names, class_to_idx

model.load_state_dict(torch.load("models/emotion_model.pth"))
model.eval()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0) 

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

PAD = 0.2  # fraction of the face box to add on each side


while True:
    ret, frame = cap.read()  
    if not ret:
        print("Error: Could not read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))  # Detect faces
    for (x, y, w, h) in faces:
        # Pad relative to the face box, then clamp to the frame so we never
        # slice with a negative index (numpy would wrap around silently).
        pad_w, pad_h = int(w * PAD), int(h * PAD)
        x1 = max(x - pad_w, 0)
        y1 = max(y - pad_h, 0)
        x2 = min(x + w + pad_w, frame.shape[1])
        y2 = min(y + h + pad_h, frame.shape[0])

        cropped_frame = frame[y1:y2, x1:x2]
        if cropped_frame.size == 0:
            continue
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        gray_crop = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
        rgb_crop = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2RGB)
        img = Image.fromarray(rgb_crop)
        tensor = test_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor)
            y_pred = torch.argmax(output, dim = 1)
            emotion = class_names[y_pred.item()]
            probs = torch.softmax(output, dim=1)[0]
        cv2.putText(frame, f'Emotion: {emotion}', (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.9, (36, 255, 12), 2)
        cv2.putText(frame, f'Confidence: {probs[y_pred.item()]:.2f}', (x, y - 40), cv2.FONT_HERSHEY_COMPLEX, 0.9, (36, 255, 12), 2)
        cv2.imshow('Face Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()