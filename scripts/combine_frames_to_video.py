import os
import cv2

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = os.getcwd()
FRAMES_DIR = os.path.join(BASE_DIR, "outputs", "video_frames")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "videos")
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "cartoon_final1.mp4")

FPS = 12

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# LOAD & SORT FRAMES (IMPORTANT)
# -----------------------------
frames = sorted(
    [f for f in os.listdir(FRAMES_DIR) if f.endswith(".png")],
    key=lambda x: int(x.split("_")[1].split(".")[0])
)

if len(frames) == 0:
    raise RuntimeError("❌ No frames found in outputs/video_frames")

print(f"✅ Found {len(frames)} frames")

# -----------------------------
# READ FIRST FRAME
# -----------------------------
first_frame_path = os.path.join(FRAMES_DIR, frames[0])
first_frame = cv2.imread(first_frame_path)

if first_frame is None:
    raise RuntimeError("❌ Failed to read first frame")

height, width, _ = first_frame.shape

# -----------------------------
# VIDEO WRITER
# -----------------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    FPS,
    (width, height)
)

# -----------------------------
# WRITE FRAMES
# -----------------------------
for frame_name in frames:
    frame_path = os.path.join(FRAMES_DIR, frame_name)
    frame = cv2.imread(frame_path)

    if frame is None:
        print(f"⚠ Skipping unreadable frame: {frame_name}")
        continue

    # Safety resize (prevents codec mismatch)
    frame = cv2.resize(frame, (width, height))
    video.write(frame)

video.release()

print("🎉 FINAL VIDEO CREATED SUCCESSFULLY")
print("📽️ Output:", OUTPUT_VIDEO)
