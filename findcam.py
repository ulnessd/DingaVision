import cv2

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.read()[0]:
        print(f"Camera {i} is available")
        cap.release()
    else:
        print(f"Camera {i} not available")
