import cv2
import torch
import numpy as np
from PIL import Image
from model import model
from data_setup import test_transform, device, class_names

model.load_state_dict(torch.load("models/emotion_model.pth"))
model.eval()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0) 

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

PAD = 0.2  # fraction of the face box to add on each side


angry_image = cv2.resize(cv2.imread('assets/emotes/angry.png', cv2.IMREAD_UNCHANGED), (500, 500))
disgust_image = cv2.resize(cv2.imread('assets/emotes/disgust.png', cv2.IMREAD_UNCHANGED), (500, 500))
fear_image = cv2.resize(cv2.imread('assets/emotes/fear.png', cv2.IMREAD_UNCHANGED), (500, 500))
happy_image = cv2.resize(cv2.imread('assets/emotes/happy.png', cv2.IMREAD_UNCHANGED), (500, 500))
sad_image = cv2.resize(cv2.imread('assets/emotes/sad.png', cv2.IMREAD_UNCHANGED), (500, 500))
surprise_image = cv2.resize(cv2.imread('assets/emotes/surprise.png', cv2.IMREAD_UNCHANGED), (500, 500))
emotes = {'angry': angry_image, 
          'disgust': disgust_image, 
          'fear': fear_image, 
          'happy': happy_image, 
          'sad': sad_image, 
          'surprise': surprise_image}

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
            y_pred = torch.argmax(output, dim=1)
            if class_names[y_pred.item()] == 'neutral':
                y_pred = torch.topk(output, 2).indices[0][1]
            emotion = class_names[y_pred.item()]
            probs = torch.softmax(output, dim=1)[0]
            conf = probs[y_pred.item()].item()

        cv2.putText(frame, f'Emotion: {emotion}', (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.9, (36, 255, 12), 2)
        cv2.putText(frame, f'Confidence: {conf:.2f}', (x, y - 40), cv2.FONT_HERSHEY_COMPLEX, 0.9, (36, 255, 12), 2)

        # Neutral deliberately has no emote, so the corner stays as plain video.
        if emotion in emotes:
            emote = emotes[emotion]
            eh, ew = emote.shape[:2]          
            alpha = emote[:, :, 3:4] / 255.0
            color = emote[:, :, :3]
            overlay = frame[0:eh, 0:ew]
            overlay[:] = ((1 - alpha) * overlay + alpha * color).astype(np.uint8)

    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()