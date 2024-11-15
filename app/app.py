import cv2
import pytesseract
import numpy as np
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)

# Initialize your video capture
cap = cv2.VideoCapture(0)

# Global variables to store the validated extracted data
final_data = {
    "name": "",
    "role": "",
    "id_number": ""
}

# Global variable to store the last n extracted values
last_extracted_data = {
    "name": "",
    "role": "",
    "id_number": ""
}

# Load and process the reference image
img = cv2.imread("../data/carnet-base2.jpeg", cv2.IMREAD_GRAYSCALE)
img_height, img_width = img.shape
img_new_height = 600
img_new_width = int(img_width * img_new_height / img_height)
img = cv2.resize(img, (img_new_width, img_new_height))

# Features
sift = cv2.SIFT_create()
kp_img, desc_img = sift.detectAndCompute(img, None)

# Feature matching
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

def validate_extracted_data(name, role, id_number):
    print(f"name: {name},role: {role}, id_number: {id_number}")
    print(last_extracted_data)
    if (name==last_extracted_data["name"] and
        role==last_extracted_data["role"] and
        id_number==last_extracted_data["id_number"]):

        flag = True
    else:
        flag = False

    last_extracted_data["name"] = name
    last_extracted_data["role"] = role
    last_extracted_data["id_number"] = id_number
            
    return flag

def get_text_from_img(img_name, img_role, img_id):
    config = "--psm 11 --oem 1"
    name = pytesseract.image_to_string(img_name, config=config + """ -c tessedit_char_blacklist=`~!@#$%^&*()—-_=+[]{}|\\;:'”",.<>?/€£¥₹©®™✓∞∑√∆`""")
    role = pytesseract.image_to_string(img_role, config=config + """ -c tessedit_char_blacklist=`~!@#$%^&*()—-_=+[]{}|\\;:'”",.<>?/€£¥₹©®™✓∞∑√∆`""")
    id_number = pytesseract.image_to_string(img_id, config=config + " -c tessedit_char_whitelist=0123456789")

    name = [x for x in name.split("\n") if len(x) > 2]
    role = [x for x in role.split("\n") if len(x) > 2]
    id_number = [x.replace(" ", "") for x in id_number.split("\n") if len(x) > 2]

    return name, role, id_number

def get_carnet(frame, sift, kp_img, desc_img, height, width):
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe, None)

    matches = flann.knnMatch(desc_img, desc_grayframe, k=2)
    good_points = []
    for m, n in matches:
        if m.distance < n.distance * 0.5:
            good_points.append(m)

    if len(good_points) > 10:
        query_pts = np.float32([kp_img[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
        train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
        h, _ = cv2.findHomography(train_pts, query_pts, cv2.RANSAC, 5.0)

        if h is not None:
            img_h = cv2.warpPerspective(grayframe, h, (width, height))
            return img_h

    return None

def get_frame():
    global final_data  # Access the global variable
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_h = get_carnet(frame, sift, kp_img, desc_img, img_new_height, img_new_width)
        if img_h is not None:
            img_name = img_h[395:455, 30:-30]
            img_role = img_h[455:485, 30:-30]
            img_id = img_h[480:520, 20:200]

            # Extract text and update the global variable
            name, role, id_number = get_text_from_img(img_name, img_role, img_id)
            name = " ".join(name) if name else "N/A"  # Get the first result or "N/A"
            role = role[0] if role else "N/A"
            id_number = id_number[0] if id_number else "N/A" 


            if validate_extracted_data(name, role, id_number):
                final_data["name"] = name
                final_data["role"] = role
                final_data["id_number"] = id_number

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/data')
def data():
    # Return the latest extracted data as JSON
    return jsonify(final_data)

if __name__ == '__main__':
    app.run(debug=True) 