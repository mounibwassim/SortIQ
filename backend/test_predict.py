import urllib.request
import json
import base64
import io
from PIL import Image

def test():
    img = Image.new('RGB', (300, 300), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    img_bytes = buf.getvalue()
    b64 = base64.b64encode(img_bytes).decode()

    # 1. Test predict-realtime
    print("Testing /predict-realtime...")
    req = urllib.request.Request(
        'http://127.0.0.1:8001/predict-realtime',
        data=json.dumps({'frame_base64': b64}).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    try:
        resp = urllib.request.urlopen(req)
        raw = resp.read().decode('utf-8')
        print("REALTIME RESPONSE OK:", raw.encode('ascii', 'xmlcharrefreplace').decode('ascii'))
    except Exception as e:
        print("REALTIME ERROR:", e)
        if hasattr(e, 'read'):
            print("DETAILS:", e.read().decode('utf-8', errors='ignore'))

    # 2. Test predict-upload
    print("\nTesting /predict-upload...")
    boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
    header_str = (
        f'--{boundary}\r\n'
        'Content-Disposition: form-data; name="file"; filename="test.jpg"\r\n'
        'Content-Type: image/jpeg\r\n\r\n'
    )
    footer_str = f'\r\n--{boundary}--\r\n'

    payload = header_str.encode('utf-8') + img_bytes + footer_str.encode('utf-8')
    req = urllib.request.Request(
        'http://127.0.0.1:8001/predict-upload',
        data=payload,
        headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}
    )
    try:
        resp = urllib.request.urlopen(req)
        raw = resp.read().decode('utf-8')
        print("UPLOAD RESPONSE OK:", raw.encode('ascii', 'xmlcharrefreplace').decode('ascii'))
    except Exception as e:
        print("UPLOAD ERROR:", e)
        if hasattr(e, 'read'):
            print("DETAILS:", e.read().decode('utf-8', errors='ignore'))

if __name__ == "__main__":
    test()
