import cv2

def parse_video(video_path):
    # Load video
    cap = cv2.VideoCapture(video_path)

    # Load custom image
    ref_img = cv2.imread("assets/15-16.png", cv2.IMREAD_GRAYSCALE)

    # ORB detector
    orb = cv2.ORB_create()

    # Compute keypoints and descriptors for reference image
    kp1, des1 = orb.detectAndCompute(ref_img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    frame_num = 0
    matches_found = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect features in frame
        kp2, des2 = orb.detectAndCompute(gray_frame, None)
        if des2 is None:
            continue

        # Match descriptors
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Heuristic: if enough good matches, consider it detected
        if len(matches) > 30:  # adjust threshold as needed
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            matches_found.append((frame_num, timestamp))
            print(f"Match found at frame {frame_num}, time {timestamp:.2f}s")

    cap.release()