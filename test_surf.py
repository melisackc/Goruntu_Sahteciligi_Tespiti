import cv2
import numpy as np
import mahotas.features.surf

def test_surf():
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (80, 80), 255, -1)
    
    spoints = mahotas.features.surf.surf(img)
    print(f"Detected {len(spoints)} points")
    if len(spoints) > 0:
        print("Descriptor shape:", spoints[:, 5:].shape)

if __name__ == "__main__":
    test_surf()
