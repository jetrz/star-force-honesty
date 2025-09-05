import numpy as np
import cv2, os, shutil


def parse_video(video_path):
    cap = cv2.VideoCapture(video_path)
    template = cv2.imread("assets/15-16.jpg", cv2.IMREAD_GRAYSCALE)

    # Use SIFT (or AKAZE if you prefer)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(template, None)

    bf = cv2.BFMatcher(cv2.NORM_L2)

    frame_num = 0
 
    # Track the best matches
    best_frames = []
    best_match_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        kp2, des2 = sift.detectAndCompute(gray, None)
        if des2 is None:
            continue

        # KNN matching
        matches = bf.knnMatch(des1, des2, k=2)

        # Lowe's ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        # Update best frame(s)
        if len(good) > best_match_count:
            best_match_count = len(good)
            best_frames = [(frame_num, frame.copy(), kp2, good)]
        elif len(good) == best_match_count and best_match_count > 0:
            # Store multiple frames if tied
            best_frames.append((frame_num, frame.copy(), kp2, good))

    cap.release()

    # Save best frames
    output_dir = "assets/matches"
    shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    for i, (fnum, frame, kp2, good) in enumerate(best_frames):
        img_matches = cv2.drawMatches(template, kp1, frame, kp2, good[:20], None, flags=2)
        out_path = os.path.join(output_dir, f"best_frame_{fnum}.jpg")
        cv2.imwrite(out_path, img_matches)
        print(f"Saved best match frame {fnum} with {best_match_count} good matches → {out_path}")

    print(f"Best match count: {best_match_count}, total best frames: {len(best_frames)}")