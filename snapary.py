import streamlit as st
from pytubefix import YouTube
from urllib.parse import urlparse, parse_qs
from moviepy.editor import VideoFileClip
from youtube_transcript_api import YouTubeTranscriptApi
import whisper
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import cv2
import os
import pandas as pd
from openai import OpenAI
import time

st.set_page_config(page_title="YouTube Video Processing", layout="wide")

# ----------------------------
# Helper functions (from your code)
# ----------------------------

def get_video_id(youtube_url):
    query = urlparse(youtube_url)
    if query.hostname == 'youtu.be':
        return query.path[1:]
    if query.hostname in ('www.youtube.com', 'youtube.com'):
        if query.path == '/watch':
            return parse_qs(query.query)['v'][0]
        elif query.path.startswith('/embed/'):
            return query.path.split('/')[2]
        elif query.path.startswith('/v/'):
            return query.path.split('/')[2]
    raise ValueError("올바른 YouTube 링크가 아닙니다.")

def download_youtube_video(url, download_audio_only=False):
    try:
        yt = YouTube(url)
        st.info(f"🎥 영상 제목: {yt.title}")
        if download_audio_only:
            stream = yt.streams.filter(only_audio=True).first()
            st.info("🎧 오디오 스트림 다운로드 중...")
        else:
            stream = yt.streams.get_highest_resolution()
            st.info("📥 영상 스트림 다운로드 중...")
        output_path = stream.download()
        st.success(f"✅ 다운로드 완료: {output_path}")
        return output_path
    except Exception as e:
        st.error(f"⚠️ 다운로드 중 오류 발생: {e}")
        return None

def extract_audio_from_video(video_path, audio_path="extracted_audio.wav"):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, codec='pcm_s16le')
    return audio_path

def get_captions(video_url, lang='en'):
    video_id = get_video_id(video_url)
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
        st.success("✅ 자막을 불러왔습니다.")
        return transcript
    except Exception as e:
        st.warning(f"❌ 자막을 불러올 수 없습니다: {e}")
        return None

def transcribe_audio_whisper(audio_path, model_size):
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)
    return result

def convert_whisper_to_caption_format(result):
    segments = result.get('segments', [])
    captions = []
    for segment in segments:
        text = segment['text'].strip()
        start = segment['start']
        end = segment['end']
        duration = round(end - start, 2)
        captions.append({'text': text, 'start': round(start, 2), 'duration': duration})
    return captions

def extract_txt(video_url, cap_lang, video_path=None, model_size="base"):
    captions = get_captions(video_url, cap_lang)
    if not captions:
        if video_path is None:
            st.error("video_path is 필요합니다. 자막 없을 때.")
            return None
        audio_path = extract_audio_from_video(video_path)
        result = transcribe_audio_whisper(audio_path, model_size=model_size)
        captions = convert_whisper_to_caption_format(result)
    return captions

def detect_and_save_scenes(video_path, threshold, output_dir="scene_images"):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    video_manager.release()
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    scene_info = []
    for i, scene in enumerate(scene_list):
        start_time = scene[0].get_seconds()
        end_time = scene[1].get_seconds()
        mid_time = (start_time + end_time) / 2
        cap.set(cv2.CAP_PROP_POS_MSEC, mid_time * 1000)
        ret, frame = cap.read()
        if ret:
            filename = f'scene_{i+1:03}.jpg'
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            scene_info.append({
                "scene": i+1,
                "start_time": start_time,
                "end_time": end_time,
                "image_path": filepath
            })
    cap.release()
    return scene_info

def split_into_chapters_df(captions, api_key, model="gpt-3.5-turbo"):
    client = OpenAI(api_key=api_key)
    text_with_time = []
    for cap in captions:
        end = cap['start'] + cap['duration']
        text_with_time.append(f"[{cap['start']:.2f} ~ {end:.2f}] {cap['text'].strip()}")
    joined_text = "\n".join(text_with_time)
    prompt = f"""
다음은 영상 자막입니다. 시간 정보를 참고해서 주요 내용을 기준으로 논리적인 챕터로 나누고,
각 챕터마다 다음 형식에 맞게 출력해주세요:

제목 | 시작시간 | 종료시간

각 항목은 반드시 '|' 기호로 구분해주세요.
시간은 초 단위 숫자(float)로 표시해주세요.

예시:
모델 소개 | 0.0 | 45.2
훈련 데이터 | 45.2 | 89.4

자막:
{joined_text}
"""
    time.sleep(1)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    result = response.choices[0].message.content.strip()
    rows = []
    for line in result.split('\n'):
        line = line.strip()
        if not line or '|' not in line:
            continue
        if '제목' in line or '시작시간' in line:
            continue
        try:
            title, start, end = [part.strip() for part in line.split('|')]
            rows.append({"chapter_title": title, "start_time": float(start), "end_time": float(end)})
        except:
            continue
    df = pd.DataFrame(rows)
    return df

def summarize_chapters(captions, chapter_df, api_key, model):
    client = OpenAI(api_key=api_key)
    summaries = []
    for _, row in chapter_df.iterrows():
        start, end = row['start_time'], row['end_time']
        segment_texts = [cap['text'].strip() for cap in captions if start <= cap['start'] <= end]
        joined_text = " ".join(segment_texts)
        prompt = f"""
다음은 하나의 영상 챕터에 해당하는 자막입니다
이 자막을 바탕으로 중요한 내용을 추려, ChatGPT가 작성하는 것처럼 보기 좋고 읽기 쉽게 정리해 주세요:
    - 문단 대신 들여쓰기, 불릿, 줄 바꿈, 이모지 활용
    - 핵심 개념과 키워드 강조, 필요시 간단 설명 추가
    - 영상 흐름과 의도 잘 드러나도록 자세히 작성
    - 명사형 종결어미 사용
    - 한국어로 작성

자막:
{joined_text}
""".strip()
        time.sleep(1)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )
            summary = response.choices[0].message.content.strip()
        except Exception as e:
            st.warning(f"요약 실패: {row['chapter_title']} ({e})")
            summary = ""
        summaries.append(summary)
    chapter_df['chapter_summary'] = summaries
    return chapter_df

# ----------------------------
# Streamlit UI
# ----------------------------

st.title("🎬 Snapary")

api_key = st.text_input("🔑 OpenAI API 키 입력", type="password")
video_url = st.text_input("📺 YouTube 영상 URL 입력")

download_audio_only = st.checkbox("오디오만 다운로드 (음성 추출용)", value=False)
caption_lang = st.selectbox("자막 언어 선택", options=["en", "ko", "ja", "es", "fr"], index=0)
whisper_model_size = st.selectbox("Whisper 모델 크기 선택", options=["tiny", "base", "small", "medium", "large"], index=1)
scene_threshold = st.slider("장면 전환 감지 임계값 (0~30)", 0.0, 30.0, 27.0)

if st.button("영상 처리 시작"):
    if not video_url or not api_key:
        st.error("YouTube URL과 OpenAI API 키를 모두 입력해주세요!")
    else:
        with st.spinner("영상 다운로드 중..."):
            video_path = download_youtube_video(video_url, download_audio_only=download_audio_only)
        if video_path is None:
            st.error("다운로드 실패!")
        else:
            with st.spinner("자막 불러오기 또는 Whisper 음성 인식 중..."):
                captions = extract_txt(video_url, caption_lang, video_path=video_path, model_size=whisper_model_size)
            if not captions:
                st.error("자막/음성 인식 실패!")
            else:
                st.success(f"자막 {len(captions)}개 문장 불러옴")

                with st.spinner("장면 감지 및 이미지 저장 중..."):
                    scenes = detect_and_save_scenes(video_path, scene_threshold)
                    st.success(f"장면 {len(scenes)}개 감지됨")

                st.write("### 장면 이미지 미리보기")
                cols = st.columns(min(len(scenes), 4))
                for i, scene in enumerate(scenes):
                    with cols[i % 4]:
                        st.image(scene['image_path'], caption=f"Scene {scene['scene']} [{scene['start_time']:.2f}s~{scene['end_time']:.2f}s]")

                with st.spinner("챕터 분할 중..."):
                    chapters_df = split_into_chapters_df(captions, api_key, model="gpt-3.5-turbo")
                    st.dataframe(chapters_df)

                with st.spinner("챕터별 요약 생성 중..."):
                    chapters_df = summarize_chapters(captions, chapters_df, api_key, model="gpt-3.5-turbo")
                    st.dataframe(chapters_df[['chapter_title', 'chapter_summary']])

                st.balloons()