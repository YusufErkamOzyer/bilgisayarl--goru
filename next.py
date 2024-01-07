import cv2
import numpy as np

def absdiff(frame1,frame2):
    diff = np.abs(frame1 - frame2)
    return diff
def threshold_custom(image, threshold_value):
    # Görüntüdeki her piksel için eşikleme uygula
    result = np.where(image >= threshold_value, 255, 0)
    print(result)
    return result.astype(np.uint8)

bg_subtractor = cv2.createBackgroundSubtractorMOG2()


cap = cv2.VideoCapture(0)

ret, first_frame = cap.read()


first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
first_frame_gray = cv2.GaussianBlur(first_frame_gray, (21, 21), 0)

# Video akışında döngü
while True:
    # Video akışından bir sonraki kareyi al
    ret, frame = cap.read(0)
    if not ret:
        break

    # Grayscale ve Görüntü Ayarlama
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray, (21, 21), 0)
    fg_mask = bg_subtractor.apply(frame)

    # Farkı hesapla
    frame_diff =absdiff(first_frame_gray, frame_gray)

    # Eşikleme (Thresholding)
    _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

    # Görüntüyü göster
    cv2.imshow('Original', frame)
    cv2.imshow('Background Subtraction', thresh)
    cv2.imshow("Foreground Mask", fg_mask)

    # Çıkış için 'q' tuşuna basılmasını bekleyin
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak

cap.release()
cv2.destroyAllWindows()