import cv2 as cv
from ultralytics import YOLO
import numpy as np

model = YOLO('yolov8n.pt') # Yolov8 nano model

cap = cv.VideoCapture('') # video pathini buraya vermelisiniz
fgbg = cv.createBackgroundSubtractorKNN(detectShadows=False)

next_id = 0
active_tracks = {}
skip = 0

roi_x, roi_y, roi_width, roi_height = 100, 0, 1000, 550 #çikti ekranındaki roi çizgilerinin kordinatları

while cap.isOpened():
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    _,fgmask = cv.threshold(fgmask, 254, 255, cv.THRESH_BINARY)

    if not ret:
        break
    roi = frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]

    cv.rectangle(frame, (roi_x, roi_y), (roi_x+roi_width, roi_y+roi_height), (255, 0, 0), 2) #Bounding box ayarları, renk ve çizgi kalınlığı ayarları

    result = model.predict(roi)
# model sonuçlarını listeye yazdırma ve tespit yapma bloğu
    boxes = result[0].boxes.xyxy.tolist() 
    classes = result[0].boxes.cls.tolist()
    names = result[0].names
    confidences = result[0].boxes.conf.tolist()
    
    for track in active_tracks.values():
        track['found'] = False

    for box, cls, conf in zip(boxes, classes, confidences):
        x1, y1, x2, y2 = map(int, box)
        x1 += roi_x
        y1 += roi_y
        x2 += roi_x
        y2 += roi_y
        confidence = conf
        detected_class = cls
        class_name = names[int(cls)]

        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        matched = False
        for track_id, track in active_tracks.items(): #tespit edilen her nesne için bir id ataması yapılması
            dist = np.linalg.norm(np.array([track['x'], track['y']]) - np.array([(x1 + x2) / 2, (y1 + y2) / 2]))
            if dist < 50:
                matched = True
                track['x'] = (x1 + x2) / 2
                track['y'] = (y1 + y2) / 2
                track['found'] = True
                cv.putText(frame, f'ID: {track_id}', (x1, y1 - 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2) #id özelliklik ayarları
                break

        if not matched:
            active_tracks[next_id] = {'x': (x1 + x2) / 2, 'y': (y1 + y2) / 2, 'found': True}
            cv.putText(frame, f'ID: {next_id}', (x1, y1 - 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            next_id += 1

        label = f'{class_name} - {confidence:.2f}'
        cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2) #bounding box class yazı tipi, renk ve yazı kalınlığı ayarları 

    for track_id, track in list(active_tracks.items()):
        if not track['found']:
            del active_tracks[track_id]    

    cv.imshow('bbox_tespit', frame) #işlenmiş video ekran çıktısı
    cv.imshow('mask',fgmask) #background mask ekran çıktısı (noisy ve bozulmaları tespit etmek için kullanıyorum)


    key = cv.waitKey(1)
    if key & 0xFF == ord('q'): # 'q' tuşu ile videodan çıkma fonksiyonu
        break
    elif key & 0xFF == ord('p'): # 'p' tuşu ile video durduma ve başlatma
        cv.waitKey(-1)
    elif key & 0xFF == ord('s'): # 's' tuşu ile videoyu 150 frame ileri sarma 
        for _ in range(150):
            ret, frame = cap.read()
            if not ret:
                break
            skip += 1
cap.release()
cv.destroyAllWindows() #bütün pencereleri kapatma 