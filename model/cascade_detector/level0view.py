import torch
import cv2
import numpy as np
from level0 import FaceDetector

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    newmodel = FaceDetector(device).to(device)
    newmodel.load_state_dict(torch.load("registry/weights/face_det.pth"))
    newmodel.eval()

    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # frame = cv2.resize(frame, (256, 256))
        
        frame = torch.from_numpy(frame.astype(np.float32)).to(device) / 255
        frame = frame.permute(2, 0, 1)
        height = frame.shape[1]
        width = frame.shape[2]
        dif = width - height
        frame = frame[:, 100:356, 100:356]
        if not ret:
            break
        
        # Detect and visualize eye areas
        result_frame = newmodel(frame)
        
        result_frame = result_frame[0]
        result_frame = result_frame.permute(1, 2, 0).cpu().detach().numpy()
        thr, result_frame = cv2.threshold(result_frame, 0.2, 1, cv2.THRESH_BINARY)
        result_frame = cv2.resize(result_frame, (512, 512))
        print(result_frame.max(), result_frame.min())

        # Display the resulting frame
        cv2.imshow('Face Area Detection', result_frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

test()