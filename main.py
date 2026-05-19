from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import anthropic
import tempfile
import os
import json
import urllib.request

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ANTHROPIC_KEY = "ANTHROPIC_KEY_HERE"
MODEL_PATH = "/tmp/pose_landmarker.task"

def get_model():
    if not os.path.exists(MODEL_PATH):
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
        urllib.request.urlretrieve(url, MODEL_PATH)
    return MODEL_PATH

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return round(angle, 1)

def analyze_golf_video(video_path):
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    model_path = get_model()

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE
    )

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    frame_indices = np.linspace(0, max(total_frames-1, 0), min(15, total_frames), dtype=int)
    all_frames = []

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                lm = result.pose_landmarks[0]

                def pt(i):
                    return [lm[i].x, lm[i].y]

                ls, rs = pt(11), pt(12)
                le, re = pt(13), pt(14)
                lw, rw = pt(15), pt(16)
                lh, rh = pt(23), pt(24)
                lk, rk = pt(25), pt(26)
                la, ra = pt(27), pt(28)
                nose = pt(0)

                ms = [(ls[0]+rs[0])/2, (ls[1]+rs[1])/2]
                mh = [(lh[0]+rh[0])/2, (lh[1]+rh[1])/2]

                all_frames.append({
                    "frame": int(idx),
                    "time": round(idx/fps, 2) if fps > 0 else 0,
                    "left_elbow": calculate_angle(ls, le, lw),
                    "right_elbow": calculate_angle(rs, re, rw),
                    "left_knee": calculate_angle(lh, lk, la),
                    "right_knee": calculate_angle(rh, rk, ra),
                    "spine": calculate_angle(nose, ms, mh),
                    "shoulder_rot": round(abs(ls[0]-rs[0])*180, 1),
                    "hip_rot": round(abs(lh[0]-rh[0])*180, 1),
                })

    cap.release()

    if not all_frames:
        return None

    return {
        "duration": round(duration, 1),
        "frames": len(all_frames),
        "max_shoulder_rot": round(max(f["shoulder_rot"] for f in all_frames), 1),
        "avg_hip_rot": round(np.mean([f["hip_rot"] for f in all_frames]), 1),
        "avg_spine": round(np.mean([f["spine"] for f in all_frames]), 1),
        "avg_left_knee": round(np.mean([f["left_knee"] for f in all_frames]), 1),
        "avg_right_elbow": round(np.mean([f["right_elbow"] for f in all_frames]), 1),
        "frames_data": all_frames
    }

GOLF_BENCHMARKS = {
    "driver": {"s": 110, "h": 45, "sp": 35, "k": 145, "e": 90, "p": "Tiger Woods, Rory McIlroy"},
    "iron": {"s": 95, "h": 40, "sp": 40, "k": 150, "e": 95, "p": "Collin Morikawa, Jon Rahm"},
    "chip": {"s": 45, "h": 15, "sp": 45, "k": 160, "e": 140, "p": "Phil Mickelson, Tiger Woods"},
    "putt": {"s": 15, "h": 5, "sp": 40, "k": 155, "e": 145, "p": "Tiger Woods, Ben Crenshaw"},
}

@app.post("/analyze/golf")
async def analyze_golf(
    video: UploadFile = File(...),
    shot_type: str = Form(default="driver"),
    goal: str = Form(default="distance")
):
    suffix = "."+video.filename.split(".")[-1] if video.filename and "." in video.filename else ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await video.read())
        tmp_path = tmp.name

    try:
        data = analyze_golf_video(tmp_path)
        if not data:
            return {"error": "Could not detect body in video. Make sure full body is visible and well lit."}

        shot_key = "driver"
        for k in ["iron", "chip", "putt"]:
            if k in shot_type.lower():
                shot_key = k
                break

        b = GOLF_BENCHMARKS[shot_key]

        prompt = (
            "You are an elite PGA golf coach analyzing REAL biomechanical data from pose detection AI.\n"
            f"Shot: {shot_type} | Goal: {goal} | Pros: {b['p']}\n\n"
            f"REAL DATA:\n"
            f"- Max shoulder rotation: {data['max_shoulder_rot']} deg (Elite: {b['s']} deg)\n"
            f"- Avg hip rotation: {data['avg_hip_rot']} deg (Elite: {b['h']} deg)\n"
            f"- Avg spine angle: {data['avg_spine']} deg (Elite: {b['sp']} deg)\n"
            f"- Avg left knee: {data['avg_left_knee']} deg (Elite: {b['k']} deg)\n"
            f"- Avg right elbow: {data['avg_right_elbow']} deg (Elite: {b['e']} deg)\n\n"
            f"Use ACTUAL numbers. Return ONLY valid JSON:\n"
            '{{"score":<0-100>,"shoulder_rotation":"' + str(data['max_shoulder_rot']) + ' deg",'
            '"hip_rotation":"' + str(data['avg_hip_rot']) + ' deg",'
            '"spine_angle":"' + str(data['avg_spine']) + ' deg",'
            '"knee_flex":"' + str(data['avg_left_knee']) + ' deg",'
            '"shoulder_rotation_class":"<good|warn|bad>",'
            '"hip_rotation_class":"<good|warn|bad>",'
            '"spine_angle_class":"<good|warn|bad>",'
            '"knee_flex_class":"<good|warn|bad>",'
            '"percentile":<0-100>,'
            '"pro_comparison":{{"m1_pct":<0-100>,"m2_pct":<0-100>,"m3_pct":<0-100>,"m4_pct":<0-100>}},'
            '"feedback":['
            '{{"type":"good","title":"<strength>","body":"<2 sentences with actual numbers>"}},'
            '{{"type":"warn","title":"<gap from elite>","body":"<drill with exact degrees to gain>"}},'
            '{{"type":"bad","title":"<biggest weakness>","body":"<exercise with measurement goal>"}}'
            '],'
            '"quick_fixes":["<fix 1 with actual angle>","<fix 2>","<fix 3>"]}}'
        )

        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )

        result = json.loads(msg.content[0].text.replace("```json", "").replace("```", "").strip())
        result["measured_data"] = data
        return result

    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass

@app.get("/health")
async def health():
    return {"status": "SportyAI Golf AI running", "version": "4.0"}
