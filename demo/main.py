import cv2
import numpy as np
import pytesseract
import urllib.parse

from sqlalchemy import URL
from sqlalchemy.sql import func
from sqlalchemy import (create_engine, 
                        MetaData, Table, Column, DateTime, Text, String,
                        insert)


# load it with enviroment variables in a .env file
password = urllib.parse.quote_plus("password") # plain (unescaped) text

url_object = URL.create(
    "postgresql+psycopg2",
    username="postgres",
    password=password,  
    host="localhost",
    port=5432,
    database="postgres",
)
engine = create_engine(url_object, echo=True)

metadata_obj = MetaData(schema="public")
carnet_detection_table = Table(
    "carnet_detection",
    metadata_obj,
    Column("datetime", DateTime(timezone=True), primary_key=True, server_default=func.now()),
    Column("name", Text),
    Column("role", Text),
    Column("id_number", String(10))
)

def insert_data(name, role, id_number):
    stmt = insert(carnet_detection_table).values(name=name,
                                                 role=role,
                                                 id_number=id_number)
    
    with engine.begin() as conn:
        conn.execute(stmt)

def get_text_from_img(img_name, img_role, img_id):
    config = "--psm 11 --oem 1"
    name = pytesseract.image_to_string(img_name, config=config+""" -c tessedit_char_blacklist=`~!@#$%^&*()—-_=+[]{}|\\;:\'",.<>?/€£¥₹©®™✓∞∑√∆`""")
    role = pytesseract.image_to_string(img_role, config=config+""" -c tessedit_char_blacklist=`~!@#$%^&*()—-_=+[]{}|\\;:'",.<>?/€£¥₹©®™✓∞∑√∆`""")
    id_number = pytesseract.image_to_string(img_id, config=config+" -c tessedit_char_whitelist=0123456789")

    name = [x for x in name.split("\n") if len(x)>2]
    role = [x for x in role.split("\n") if len(x)>2]
    id_number = [x.replace(" ", "") for x in id_number.split("\n") if len(x)>2] 

    return name, role, id_number

def get_carnet(frame, sift, kp_img, desc_img, height, width):
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # trainimage
    kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe, None)

    matches = flann.knnMatch(desc_img, desc_grayframe, k=2)
    good_points = []
    for m, n in matches:
        if m.distance < n.distance*0.5:
            good_points.append(m)

    if len(good_points)>10:
        query_pts = np.float32([kp_img[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
        train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
        h, _ = cv2.findHomography(train_pts, query_pts, cv2.RANSAC, 5.0)        

        # Perspective transform
        if h is not None:
            img_h = cv2.warpPerspective(grayframe, h, (width, height))
            cv2.imshow("Extracted", img_h)
        
            return img_h
        
    else:
        cv2.imshow("Homography", grayframe)
        return None


if __name__ == "__main__":
    img = cv2.imread("../data/carnet-base2.jpeg", cv2.IMREAD_GRAYSCALE)  # queryiamge
    img_height, img_width = img.shape
    img_new_height = 600
    img_new_width = int(img_width*img_new_height/img_height)
    img = cv2.resize(img, (img_new_width, img_new_height))


    # Features
    sift = cv2.SIFT_create()
    kp_img, desc_img = sift.detectAndCompute(img, None)

    # Feature matching
    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)


    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        img_h = get_carnet(frame, sift, kp_img, desc_img, img_new_height, img_new_width)
        if img_h is not None:
            img_name = img_h[395:455,30:-30]
            img_role = img_h[455:485,30:-30]
            img_id = img_h[480:520,20:200]

            cv2.imshow("Name", img_name)
            cv2.imshow("Role", img_role)
            cv2.imshow("ID Number", img_id)

            name, role, id_number = get_text_from_img(img_name, img_role, img_id)
            print("-"*30)
            print(name)
            print(role)
            print(id_number)

            # insert_data(name, role, id_number)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# API Flask o FastAPI
# UI - Imagen y datos extraidos
#    - Envio automático, validar 
#    - Luz verde para avanzar
# Contenedor
