import base64
import time
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
from ultralytics import YOLO

try:
    import resend
except ImportError:
    resend = None


# ============================================================
# 基本設定
# ============================================================
OUTPUT_DIR = Path("fire_detection_output")
SCREENSHOT_DIR = OUTPUT_DIR / "fire_screenshots"
OUTPUT_DIR.mkdir(exist_ok=True)
SCREENSHOT_DIR.mkdir(exist_ok=True)

MODEL_PATH = "fire_smoke_best.pt"  # 已預先下載好放在repo根目錄
DEMO_VIDEOS = {
    "示範影片 1 ": "assets/input_video_1.mp4",
    "示範影片 2 ": "assets/input_video_2.mp4",
    "示範影片 3 ": "assets/input_video_3.mp4",
}

ALERT_COOLDOWN_SEC = 60
CONFIDENCE_THRESHOLD = 0.3
FRAME_SKIP = 6          # 每N幀辨識一次，兼顧速度與流暢度
MAX_FRAMES = 300         # 公開demo限制最長處理幀數，避免免費方案資源被單次請求佔滿


# ============================================================
# 資源載入（快取，避免每次互動都重新載入模型）
# ============================================================
@st.cache_resource(show_spinner="正在載入YOLOv8火災辨識模型...")
def load_model():
    return YOLO(MODEL_PATH)


def get_resend_key():
    """優先讀取 st.secrets，本機開發時退而求其次讀環境變數"""
    try:
        return st.secrets["RESEND_API_KEY"]
    except Exception:
        import os
        return os.environ.get("RESEND_API_KEY")


def get_alert_recipient():
    try:
        return st.secrets["ALERT_EMAIL_TO"]
    except Exception:
        import os
        return os.environ.get("ALERT_EMAIL_TO", "")


# ============================================================
# 告警信（含截圖附件），有冷卻時間避免demo被灌爆
# ============================================================
def send_fire_alert(frame_count, confidence, video_time_sec, screenshot_path):
    api_key = get_resend_key()
    recipient = get_alert_recipient()

    if not api_key or not recipient or resend is None:
        return False, "尚未設定 RESEND_API_KEY / ALERT_EMAIL_TO，僅記錄偵測結果，不寄送告警信"

    now = time.time()
    last_alert = st.session_state.get("last_alert_time", 0)
    if now - last_alert < ALERT_COOLDOWN_SEC:
        remaining = ALERT_COOLDOWN_SEC - (now - last_alert)
        return False, f"告警冷卻中，尚需 {remaining:.0f} 秒後才會再次寄信"

    resend.api_key = api_key
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        with open(screenshot_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")

        params = {
            "from": "Acme <onboarding@resend.dev>",
            "to": [recipient],
            "subject": "🔥 火災偵測系統告警通知（線上Demo）",
            "html": f"""
                <h2 style="color:#d32f2f;">⚠️ 偵測到火災 (Fire) ⚠️</h2>
                <p><strong>偵測時間：</strong>{timestamp}</p>
                <p><strong>影片時間：</strong>{video_time_sec:.2f} 秒（第 {frame_count} 幀）</p>
                <p><strong>信心度：</strong>{confidence*100:.2f}%</p>
                <p>詳見附件截圖。（本信件由公開展示Demo觸發）</p>
            """,
            "attachments": [
                {"filename": Path(screenshot_path).name, "content": image_base64}
            ],
        }
        resend.Emails.send(params)
        st.session_state["last_alert_time"] = now
        return True, "已寄出告警信（含截圖附件）"
    except Exception as e:
        return False, f"告警信寄送失敗：{e}"


# ============================================================
# 主辨識流程
# ============================================================
def run_detection(video_path, model, frame_placeholder, progress_bar, status_text):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        st.error("無法開啟影片檔案")
        return None

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    process_limit = min(total_frames, MAX_FRAMES)
    truncated = total_frames > MAX_FRAMES

    detection_log = []
    fire_frames, smoke_frames = [], []
    frame_count = 0
    alert_messages = []

    while frame_count < process_limit:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if frame_count % FRAME_SKIP != 0:
            continue

        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        annotated = results[0].plot()

        detections = results[0].boxes
        fire_hit = False
        max_fire_conf = 0.0

        for box in detections:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id]
            bbox = box.xyxy[0].cpu().numpy().tolist()

            detection_log.append({
                "時間戳記": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "幀編號": frame_count,
                "時間(秒)": round(frame_count / fps, 2),
                "檢測類別": class_name,
                "信心度(%)": round(conf * 100, 2),
                "邊界框": bbox,
            })

            if class_name.lower() == "fire":
                fire_frames.append(frame_count)
                fire_hit = True
                max_fire_conf = max(max_fire_conf, conf)
            elif class_name.lower() == "smoke":
                smoke_frames.append(frame_count)

        if fire_hit:
            screenshot_name = f"fire_frame{frame_count}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            screenshot_path = SCREENSHOT_DIR / screenshot_name
            cv2.imwrite(str(screenshot_path), annotated)

            ok, msg = send_fire_alert(frame_count, max_fire_conf, frame_count / fps, screenshot_path)
            alert_messages.append(("success" if ok else "info", msg))

        # 更新畫面（RGB顯示）
        frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")
        progress_bar.progress(min(frame_count / process_limit, 1.0))
        status_text.text(f"處理中... 第 {frame_count}/{process_limit} 幀")

    cap.release()

    return {
        "detection_log": detection_log,
        "fire_frames": sorted(set(fire_frames)),
        "smoke_frames": sorted(set(smoke_frames)),
        "fps": fps,
        "alert_messages": alert_messages,
        "truncated": truncated,
        "total_frames": total_frames,
        "processed_frames": process_limit,
    }


# ============================================================
# 畫面
# ============================================================
st.set_page_config(page_title="火災辨識系統 Demo", page_icon="🔥", layout="centered")

st.title("🔥 火災/煙霧即時辨識系統")
st.caption("YOLOv8 火災辨識 + Resend Email 告警通知｜展示Demo")

with st.expander("這個Demo在做什麼？", expanded=False):
    st.markdown(
        """
本專案用 **YOLOv8** 對影片逐幀辨識火災（fire）與煙霧（smoke）。
偵測到 `fire` 時，系統會：
1. 擷取當下標註畫面存成截圖
2. 透過 **Resend Email API** 寄出含截圖附件的告警信（60秒冷卻，避免重複寄信）

為了避免公開Demo被濫用（YouTube下載不穩定、任意輸入可能塞爆信箱），
這裡固定使用預先準備好的示範影片，也可以上傳你自己的短片測試辨識效果（僅記錄結果，不寄信）。

本機版本（USB攝影機即時串流整合，含LINE Bot通知與系統警告音）已完成開發，
但因需要實體攝影機、不適合公開部署，此次線上Demo僅提供影片辨識版本。
本機版程式碼與示範影片請洽作者索取，或見完整專案repo。
        """
    )

source_choice = st.radio(
    "選擇影片來源",
    options=list(DEMO_VIDEOS.keys()) + ["上傳我自己的影片（僅測試辨識，不寄告警信）"],
)

uploaded_file = None
video_path = None

if source_choice == "上傳我自己的影片（僅測試辨識，不寄告警信）":
    uploaded_file = st.file_uploader("上傳影片檔（mp4，建議30秒內、720p以下）", type=["mp4", "mov", "avi"])
    if uploaded_file:
        video_path = OUTPUT_DIR / "user_upload.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())
else:
    video_path = Path(DEMO_VIDEOS[source_choice])
    if not video_path.exists():
        st.warning(f"找不到示範影片檔案：{video_path}（部署時請確認 assets/ 資料夾已包含此檔案）")
        video_path = None

is_upload_mode = source_choice.startswith("上傳")

run_btn = st.button("▶ 開始辨識", type="primary", disabled=video_path is None)

if run_btn and video_path is not None:
    model = load_model()

    frame_placeholder = st.empty()
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    # 上傳模式暫時關閉寄信，避免陌生使用者濫用Email額度
    original_key = None
    if is_upload_mode:
        st.session_state["_upload_mode_active"] = True

    with st.spinner("辨識進行中，請稍候..."):
        result = run_detection(video_path, model, frame_placeholder, progress_bar, status_text)

    status_text.empty()
    progress_bar.empty()

    if result is None:
        st.error("辨識失敗，請確認影片檔案是否正確")
    else:
        st.success("✅ 辨識完成！")

        if result["truncated"]:
            st.warning(
                f"⚠️ 這支影片共有 {result['total_frames']} 幀，超過本Demo單次處理上限"
                f"（{MAX_FRAMES} 幀，約 {MAX_FRAMES / result['fps']:.0f} 秒），"
                f"僅分析了前 {result['processed_frames']} 幀，後段內容未被辨識。"
                "建議上傳較短的片段以取得完整結果。"
            )

        col1, col2, col3 = st.columns(3)
        col1.metric("總檢測次數", len(result["detection_log"]))
        col2.metric("火災 (fire) 幀數", len(result["fire_frames"]))
        col3.metric("煙霧 (smoke) 幀數", len(result["smoke_frames"]))

        if is_upload_mode:
            st.info("上傳影片測試模式：僅顯示辨識結果，不會實際寄送告警信。完整寄信功能可在下方示範影片體驗。")
        else:
            for level, msg in result["alert_messages"]:
                getattr(st, level)(f"📧 {msg}")
            if not result["alert_messages"] and len(result["fire_frames"]) == 0:
                st.info("這段影片未偵測到火災，因此沒有觸發告警信。")

        if result["detection_log"]:
            df = pd.DataFrame(result["detection_log"])
            st.subheader("偵測紀錄")
            st.dataframe(df, use_container_width=True)

            csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("下載CSV日誌", csv_bytes, "detection_log.csv", "text/csv")
        else:
            st.info("這段影片中未偵測到任何火災或煙霧。")

st.divider()
st.caption("完整程式碼與本機USB攝影機即時監控版本，請見 GitHub repo README。")
