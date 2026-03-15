
import base64
import requests  # pyre-ignore
import cv2  # pyre-ignore
import numpy as np  # pyre-ignore
import time

def test_case(name, image_path=None, dummy_type='face'):
    print(f"\n--- Testing: {name} ---")
    
    # Create dummy images for testing logic if no path provided
    if image_path:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
    else:
        # Create a dummy image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        if dummy_type == 'face':
            # Create a skin-tone block (HSV approx: H=12, S=150, V=200 -> BGR)
            # This should trigger the skin_ratio > 0.35 check
            skin_color = [100, 150, 200] # BGR approximation of skin
            img[:, :] = skin_color
        elif dummy_type == 'chair':
             cv2.rectangle(img, (200, 100), (400, 300), (50, 50, 50), -1)
        
        _, buffer = cv2.imencode('.jpg', img)
        img_b64 = base64.b64encode(buffer).decode()

    payload = {"frame_base64": f"data:image/jpeg;base64,{img_b64}"}
    
    try:
        # Note: We simulate the headers that Home.tsx sends
        headers = {
            "X-API-KEY": "test_key",
            "X-Color-Glass": "#22c55e",
            "X-Color-Plastic": "#3b82f6",
            "X-Color-Metal": "#eab308",
            "X-Color-Paper": "#f97316"
        }
        
        # We need to wait for the model to load in the background uvicorn
        print("Waiting for backend to be ready...")
        time.sleep(5) 

        resp = requests.post("http://127.0.0.1:8000/predict-realtime", json=payload, headers=headers)
        if resp.status_code == 200:
            data = resp.json()
            print(f"Summary: {data.get('summary')}")
            for det in data.get('detections', []):
                print(f"  Result: {det['label']} | Conf: {det['confidence']:.2f} | Waste: {det['is_waste']} | Color: {det['box_color_hex']}")
        else:
            print(f"Error {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    # Wait for uvicorn to be ready (it should be running in background)
    time.sleep(2)
    test_case("Face (Simulated Skin Tone)", dummy_type='face')
    test_case("Chair (Simulated Non-waste)", dummy_type='chair')
