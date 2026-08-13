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


while True:
    ret, frame = cap.read()  
    if not ret:
        print("Error: Could not read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))  # Detect faces
    for (x, y, w, h) in faces:
        cropped_frame = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        rgb_crop = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_crop)
        tensor = test_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor)
            y_pred = torch.argmax(output, dim = 1)
            emotion = class_names[y_pred.item()]
    
        cv2.putText(frame, f'Emotion: {emotion}', (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.9, (36, 255, 12), 2)
        cv2.imshow('Face Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()