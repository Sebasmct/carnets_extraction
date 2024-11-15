import cv2
import numpy as np
import time

img = cv2.imread("carnet-base2.jpeg", cv2.IMREAD_GRAYSCALE)  # queryiamge
img_height, img_width = img.shape
img_new_height = 600
img_resize_factor = img_new_height/img_height
img = cv2.resize(img, (int(img_width*img_resize_factor), img_new_height))

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Features
sift = cv2.SIFT_create()
kp_img, desc_img = sift.detectAndCompute(img, None)
# img = cv2.drawKeypoints(img, kp_img, img)

# Feature matching
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

# frame_rate = 30
# prev_time = 0
while True:
    # elapsed_time = time.time() - prev_time
    ret, frame = cap.read()

    # if elapsed_time > 1./frame_rate:
    prev = time.time()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # trainimage
    kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe, None)
    # grayframe = cv2.drawKeypoints(grayframe, kp_grayframe, grayframe)

    matches = flann.knnMatch(desc_img, desc_grayframe, k=2)
    good_points = []
    for m, n in matches:
        if m.distance < n.distance*0.5:
            good_points.append(m)
    # img3 = cv2.drawMatches(img, kp_img, grayframe, kp_grayframe, good_points, None)
    # cv2.imshow("Matches", img3)

    print(len(good_points))
    if len(good_points)>10:
        query_pts = np.float32([kp_img[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
        train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
        # h, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
        h, mask = cv2.findHomography(train_pts, query_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()        

        # Perspective transform
        height, width = img.shape
        pts = np.float32([[0, 0], [0, height], [width, height], [width, 0]]).reshape(-1, 1, 2)
        if h is not None:
            # dst = cv2.perspectiveTransform(pts, h)
            # homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
            # cv2.imshow("Homography", homography)

            img_h = cv2.warpPerspective(frame, h, (width, height))
            img_h = cv2.resize(img_h, (374, 600))
            cv2.imshow("Extracted", img_h)
        
    else:
        cv2.imshow("Homography", grayframe)

    # img = cv2.resize(img, (374, 600))
    # cv2.imshow("carnet", img)
    # frame = cv2.resize(grayframe, (600, 400))
    # cv2.imshow("PC Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# OCR
# Postgres Nombre, Cargo, Cédula