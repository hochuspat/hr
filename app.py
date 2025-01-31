import streamlit as st
import cv2
import numpy as np
import speech_recognition as sr
import librosa
import os
import dlib
from deepface import DeepFace
import logging
import requests
import ffmpeg
import traceback
from werkzeug.utils import secure_filename
import uuid
import time
from scipy.spatial import distance
import mediapipe as mp  # Добавляем MediaPipe
import PyPDF2
import re  # Для анонимизации текста

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Вывод в консоль
        logging.FileHandler("app.log", encoding='utf-8')  # Запись в файл
    ]
)
logger = logging.getLogger(__name__)

# Настройки GigaChat
GIGACHAT_AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
GIGACHAT_API_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
AUTHORIZATION_KEY = "MWJmMWU3ZDQtYTQ0NS00NGFjLTg1OGEtNGFjYmIyNjcxN2Y5OmJhYjhlYTVhLWYwMmUtNGEyOC04NjUzLTQ3MTA3OTE3YmFmMA=="  # Ваш Authorization Key из Base64

# Загрузка предобученных моделей dlib
detector = dlib.get_frontal_face_detector()
predictor_path = "shape_predictor_68_face_landmarks.dat"
if not os.path.exists(predictor_path):
    st.error(f"Не найден файл предсказателя формы: {predictor_path}. Пожалуйста, загрузите его из официального источника dlib.")
    logger.error(f"Не найден файл предсказателя формы: {predictor_path}.")
    st.stop()
predictor = dlib.shape_predictor(predictor_path)

# Инициализация MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                   max_num_faces=1,
                                   refine_landmarks=True,
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Функция для анонимизации текста
def anonymize_text(text):
    """
    Удаляет или заменяет личные данные в тексте.
    """
    try:
        # Удаление телефонных номеров
        text = re.sub(r'\+7\s*\(\d{3}\)\s*\d{3}-\d{4}', '[Телефон удалён]', text)
        # Удаление email
        text = re.sub(r'\S+@\S+', '[Email удалён]', text)
        # Удаление ссылок
        text = re.sub(r'http\S+', '[Ссылка удалена]', text)
        # При необходимости удалите или замените имена
        # Например:
        text = re.sub(r'Халилова Лия Жаудатовна', '[Имя удалено]', text)
        return text
    except Exception as e:
        logger.error(f"Ошибка при анонимизации текста: {e}")
        return text  # Возвращаем оригинальный текст в случае ошибки

# Функция для получения токена доступа от GigaChat API
def get_access_token():
    """Получает токен доступа от GigaChat API."""
    try:
        logger.info("Запрос токена доступа от GigaChat API...")
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json',
            'RqUID': str(uuid.uuid4()),  # Генерация уникального UUID
            'Authorization': f'Basic {AUTHORIZATION_KEY}',
        }
        payload = {'scope': 'GIGACHAT_API_PERS'}
        response = requests.post(GIGACHAT_AUTH_URL, headers=headers, data=payload, verify=False)

        # Логируем ответ
        logger.info(f"Ответ на запрос токена: {response.status_code} - {response.text}")

        if response.status_code == 200:
            token = response.json().get("access_token")
            if token:
                logger.info("Токен успешно получен.")
                return token
            else:
                logger.error("Токен не найден в ответе.")
                return None
        else:
            logger.error(f"Ошибка при получении токена: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка запроса к GigaChat API: {e}")
        return None
    except Exception as e:
        logger.error(f"Неизвестная ошибка: {e}")
        return None

# Функция для генерации детализированного отчета с помощью GigaChat API
def generate_analysis_report(text, token):
    """Генерирует детализированный отчет на основе анализа текста с использованием GigaChat API."""
    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        prompt = (
            "Используя приведенные ниже результаты анализа кандидата, составь детализированный отчет в соответствии с следующим шаблоном:\n\n"
            "1. Вербальные навыки:\n"
            "   - Говорение и ясность изложения:\n"
            "   - Интонация и темп речи:\n"
            "   - Уровень формальности:\n"
            "   - Аргументация:\n"
            "2. Невербальные навыки:\n"
            "   - Язык тела:\n"
            "   - Контакт глазами:\n"
            "   - Эмоции и реакция:\n"
            "   - Мимика:\n"
            "3. Софт-скилы (мягкие навыки):\n"
            "   - Эмоциональный интеллект:\n"
            "   - Кооперативность и работа в команде:\n"
            "   - Лидерские качества:\n"
            "   - Конфликтное поведение:\n"
            "4. Профессиональные качества:\n"
            "   - Цели и амбиции:\n"
            "   - Проблемное мышление:\n"
            "   - Гибкость и адаптивность:\n"
            "5. Поведение и стиль общения:\n"
            "   - Уверенность и самооценка:\n"
            "   - Открытость и доступность:\n"
            "   - Гибкость в подходах:\n"
            "6. Культурная совместимость:\n"
            "   - Ценности и убеждения:\n"
            "   - Согласованность с корпоративной культурой:\n"
            "7. Стрессоустойчивость:\n"
            "   - Реакция на стрессовые вопросы:\n"
            "8. Мотивация и драйв:\n"
            "   - Мотивация для работы:\n\n"
            "9. Тип личности и описание ENTJ, INFJ, ENFP, ISTJ, ENTP, INFP, ESFJ, ISTP:\n"
            "   - Тип личности:\n\n"
            "Вот результаты анализа:\n\n"
            f"{text}\n\n"
            "Заполни каждую секцию шаблона соответствующей информацией на основе предоставленных данных."
        )
        payload = {
            "model": "GigaChat",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,  # Можно настроить по необходимости
            "max_tokens": 1500,   # Увеличено для детализированного отчета
        }

        logger.info("Отправка запроса к GigaChat API для генерации отчета...")
        logger.info(f"Отправляемый промт:\n{prompt}")
        response = requests.post(GIGACHAT_API_URL, json=payload, headers=headers, verify=False)
        logger.info(f"Ответ от GigaChat API: {response.status_code} - {response.text}")

        if response.status_code == 200:
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                finish_reason = data["choices"][0].get("finish_reason", "")
                if finish_reason == "blacklist":
                    logger.warning("Запрос был отклонён политиками безопасности API.")
                    return "Ваш запрос был отклонён политиками безопасности API. Попробуйте изменить формулировку."
                else:
                    report = data["choices"][0]["message"]["content"].strip()
                    logger.info("Отчет успешно сгенерирован.")
                    return report
            else:
                logger.error("Ответ API не содержит ожидаемых данных.")
                return "Не удалось получить отчет от API."
        else:
            logger.error(f"Ошибка API: {response.status_code} - {response.text}")
            return f"Ошибка API: {response.status_code} - {response.text}"
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка запроса к GigaChat API: {e}")
        return f"Ошибка запроса: {e}"
    except Exception as e:
        logger.error(f"Неизвестная ошибка при генерации отчета: {e}")
        return f"Неизвестная ошибка: {e}"

# Функция для генерации оценки резюме с помощью GigaChat API
def generate_resume_evaluation(text, token):
    """Генерирует оценку резюме на основе предоставленного текста с использованием GigaChat API."""
    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        prompt = (
            "На основе предоставленного резюме составьте краткую оценку кандидата, включая следующие аспекты:\n\n"
            "1. Основные навыки и компетенции:\n"
            "2. Опыт работы и достижения:\n"
            "3. Образование и квалификации:\n"
            "4. Потенциал для развития:\n"
            "5. Рекомендации по найму:\n\n"
            "Вот резюме кандидата:\n\n"
            f"{text}\n\n"
            "Пожалуйста, заполните каждый раздел на основе предоставленного резюме."
        )
        payload = {
            "model": "GigaChat",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.5,  # Можно настроить по необходимости
            "max_tokens": 1000,   # Увеличено для детализированной оценки
        }

        logger.info("Отправка запроса к GigaChat API для генерации оценки резюме...")
        logger.info(f"Отправляемый промт для оценки резюме:\n{prompt}")
        response = requests.post(GIGACHAT_API_URL, json=payload, headers=headers, verify=False)
        logger.info(f"Ответ от GigaChat API (оценка резюме): {response.status_code} - {response.text}")

        if response.status_code == 200:
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                finish_reason = data["choices"][0].get("finish_reason", "")
                if finish_reason == "blacklist":
                    logger.warning("Запрос оценки резюме был отклонён политиками безопасности API.")
                    return "Ваш запрос оценки резюме был отклонён политиками безопасности API. Попробуйте изменить формулировку."
                else:
                    evaluation = data["choices"][0]["message"]["content"].strip()
                    logger.info("Оценка резюме успешно сгенерирована.")
                    return evaluation
            else:
                logger.error("Ответ API не содержит ожидаемых данных для оценки резюме.")
                return "Не удалось получить оценку резюме от API."
        else:
            logger.error(f"Ошибка API при генерации оценки резюме: {response.status_code} - {response.text}")
            return f"Ошибка API при генерации оценки резюме: {response.status_code} - {response.text}"
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка запроса к GigaChat API при генерации оценки резюме: {e}")
        return f"Ошибка запроса: {e}"
    except Exception as e:
        logger.error(f"Неизвестная ошибка при генерации оценки резюме: {e}")
        return f"Неизвестная ошибка: {e}"

# Координаты точек лица (dlib 68 landmarks)
LEFT_EYE_POINTS = list(range(36, 42))
RIGHT_EYE_POINTS = list(range(42, 48))
MOUTH_POINTS = list(range(48, 68))

# Функция расчета отношения сторон глаза (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Функция для определения направления взгляда
def get_gaze_direction(landmarks, frame_width, frame_height):
    """
    Определяет направление взгляда: 'left', 'right', 'center'
    на основе координат ключевых точек лица.
    """
    try:
        # Координаты зрачков (примерные индексы, могут потребовать корректировки)
        left_pupil = landmarks[468]  # Примерный индекс для зрачка левого глаза
        right_pupil = landmarks[473]  # Примерный индекс для зрачка правого глаза

        # Перевод координат из нормализованных значений в пиксели
        left_pupil_x = int(left_pupil.x * frame_width)
        left_pupil_y = int(left_pupil.y * frame_height)
        right_pupil_x = int(right_pupil.x * frame_width)
        right_pupil_y = int(right_pupil.y * frame_height)

        # Центр экрана
        center_x = frame_width / 2

        # Среднее положение зрачков
        avg_pupil_x = (left_pupil_x + right_pupil_x) / 2

        # Определение направления взгляда
        threshold = frame_width * 0.05  # 5% от ширины кадра
        if avg_pupil_x < center_x - threshold:
            return "left"
        elif avg_pupil_x > center_x + threshold:
            return "right"
        else:
            return "center"
    except IndexError:
        logger.warning("Недостаточно ключевых точек для определения направления взгляда.")
        return "unknown"

# Функция анализа видео с использованием MediaPipe
def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Не удалось открыть видео файл: {video_path}")
        return {}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_duration = total_frames / fps

    # Пропускаем кадры для ускорения анализа
    frame_skip = max(fps // 4, 5)  # Анализировать 4 кадра в секунду

    stats = {
        'smiles': 0,
        'blinks': 0,
        'stress': 0,
        'movement': 0,
        'emotions': {},
        'eye_contact': {
            'steady_eye_contact': 0,
            'frequent_gaze_averts': 0
        }
    }

    frame_count = 0
    processed_frames = 0
    last_frame_gray = None  # Для вычисления движения
    blink_counter = 0
    blink_threshold = 0.2  # Порог для определения моргания
    mouth_open_threshold = 0.4  # Порог для определения улыбки

    progress_bar = st.progress(0)
    status_text = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        processed_frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Обработка изображения с помощью MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                gaze_direction = get_gaze_direction(face_landmarks.landmark, frame.shape[1], frame.shape[0])

                if gaze_direction in ["left", "right"]:
                    stats['eye_contact']['frequent_gaze_averts'] += 1
                elif gaze_direction == "center":
                    stats['eye_contact']['steady_eye_contact'] += 1

        faces = detector(gray)

        if not faces:
            logger.debug(f"Лицо не обнаружено на кадре {frame_count}.")
            continue  # Если лица не обнаружены, пропускаем кадр
        for face in faces:
            landmarks = predictor(gray, face)

            # Извлекаем координаты лица
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()

            # Вырезаем лицо из кадра
            cropped_face = frame[y1:y2, x1:x2]

            # Проверяем, что вырезанное лицо имеет допустимые размеры
            if cropped_face.size == 0:
                logger.warning(f"Вырезанное лицо на кадре {frame_count} имеет нулевой размер.")
                continue
            # EAR для моргания
            left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in LEFT_EYE_POINTS]
            right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in RIGHT_EYE_POINTS]

            left_EAR = eye_aspect_ratio(left_eye)
            right_EAR = eye_aspect_ratio(right_eye)
            avg_EAR = (left_EAR + right_EAR) / 2.0

            if avg_EAR < blink_threshold:  # Если глаза закрылись
                blink_counter += 1
            elif blink_counter > 0:
                stats['blinks'] += 1  # Засчитываем моргание
                blink_counter = 0

            # Определение улыбки (анализ высоты рта)
            mouth = [(landmarks.part(n).x, landmarks.part(n).y) for n in MOUTH_POINTS]
            mouth_height = distance.euclidean(mouth[3], mouth[9])  # Расстояние между губами
            mouth_width = distance.euclidean(mouth[0], mouth[6])  # Ширина рта
            smile_ratio = mouth_height / mouth_width

            if smile_ratio > mouth_open_threshold:
                stats['smiles'] += 1

            # Анализ эмоций (каждый 5-й кадр для большей точности)
            if processed_frames % 5 == 0:
                try:
                    analysis = DeepFace.analyze(cropped_face, actions=['emotion'], enforce_detection=True, silent=True)
                    if analysis and isinstance(analysis, list) and len(analysis) > 0:
                        emotion = analysis[0]['dominant_emotion']  # Извлекаем первый элемент списка
                        if emotion:
                            stats['emotions'][emotion] = stats['emotions'].get(emotion, 0) + 1

                            # Если эмоция указывает на стресс
                            if emotion in ['angry', 'fear', 'sad']:
                                stats['stress'] += 1
                    else:
                        logger.warning(f"Не удалось проанализировать эмоции на кадре {frame_count}: анализ не вернул данных.")
                except Exception as e:
                    logger.warning(f"Не удалось проанализировать эмоции на кадре {frame_count}: {e}")

            # Анализ движения
            if last_frame_gray is not None:
                frame_diff = cv2.absdiff(last_frame_gray, gray)
                movement_score = np.sum(frame_diff) / frame_diff.size  # Средняя разница между кадрами

                if movement_score > 5:  # Эмпирический порог
                    stats['movement'] += 1

            last_frame_gray = gray  # Обновляем предыдущий кадр

        # Обновление прогресс-бара
        progress = min(100, int((frame_count / total_frames) * 100))
        progress_bar.progress(progress)
        status_text.text(f"Обработано {progress}% видео")

    cap.release()
    logger.info(f"Анализ видео завершен. Обработано {processed_frames} кадров.")
    return stats

# Функция для преобразования видео в аудио
def extract_audio(video_path, audio_path):
    try:
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(stream, audio_path, format='wav', acodec='pcm_s16le', ar='16000')
        ffmpeg.run(stream, overwrite_output=True)
    except Exception as e:
        logger.error(f"Ошибка при извлечении аудио: {e}")

# Функция для транскрибации аудио
def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio, language="ru-RU")
            return text
        except sr.UnknownValueError:
            return "Речь не распознана"
        except sr.RequestError:
            return "Ошибка сервиса распознавания речи"

# Функция для анализа аудио
def analyze_audio(audio_path):
    y, sr_ = librosa.load(audio_path, sr=16000)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr_)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr_))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr_))
    speech_rate = "Быстрая" if tempo > 120 else "Медленная"
    tonality = "Низкий" if spectral_centroid < 1000 else "Высокий"
    timbre = "Тонкий" if spectral_bandwidth < 500 else "Грубый"
    pause_freq = "Частые паузы" if np.mean(librosa.effects.split(y, top_db=30)) > 10 else "Редкие паузы"
    filler_sounds = "Частое использование заполняющих звуков" if np.mean(librosa.effects.preemphasis(y)) > 0.5 else "Редкое использование заполняющих звуков"

    return {
        'tonality': tonality,
        'timbre': timbre,
        'speech_rate': speech_rate,
        'pause_frequency': pause_freq,
        'filler_sounds': filler_sounds
    }

# Функция для генерации оценки резюме
def process_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        logger.info("Текст успешно извлечен из файла.")
    except Exception as e:
        logger.error(f"Ошибка извлечения текста из файла: {e}")
        return {"error": "Не удалось извлечь текст из файла"}

    # Получение токена GigaChat
    token = get_access_token()
    if not token:
        return {"error": "Не удалось получить токен GigaChat"}

    # Генерация детализированного отчета
    report = generate_analysis_report(text, token)
    if not report:
        return {"error": "Ошибка генерации отчета от GigaChat"}

    # Генерация оценки резюме
    evaluation = generate_resume_evaluation(text, token)
    if not evaluation:
        return {"error": "Ошибка генерации оценки резюме от GigaChat"}

    # Успешный результат
    return {"report": report, "evaluation": evaluation}

def process_file(file, file_type):
    """
    Обрабатывает загруженный файл и извлекает из него текст.
    
    :param file: Загруженный файл (объект BytesIO).
    :param file_type: Тип файла ('txt' или 'pdf').
    :return: Извлеченный текст или ошибка.
    """
    try:
        if file_type == 'txt':
            # Чтение текста из .txt файла
            text = file.read().decode('utf-8')
            logger.info("Текст успешно извлечен из TXT файла.")
            return {"text": text}
        
        elif file_type == 'pdf':
            # Чтение текста из .pdf файла с помощью PyPDF2
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            if text:
                # Анонимизация извлечённого текста
                anonymized_text = anonymize_text(text)
                logger.info("Текст успешно извлечен и анонимизирован из PDF файла.")
                return {"text": anonymized_text}
            else:
                logger.error("Не удалось извлечь текст из PDF файла.")
                return {"error": "Не удалось извлечь текст из PDF файла."}
        
        else:
            logger.error(f"Не поддерживаемый тип файла: {file_type}")
            return {"error": f"Не поддерживаемый тип файла: {file_type}"}
    
    except Exception as e:
        logger.error(f"Ошибка при обработке файла: {e}")
        return {"error": f"Ошибка при обработке файла: {e}"}

# Интерфейс Streamlit
st.title("Многофункциональный Анализ Кандидатов")

# Создаем вкладки для разделения функциональности
tabs = st.tabs(["Анализ Видео/Аудио", "Анализ Текстовых и PDF Файлов"])

# Вкладка 1: Анализ Видео/Аудио
with tabs[0]:
    st.header("Анализ видео и аудио кандидата")

    uploaded_video = st.file_uploader("Загрузите видео файл", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        try:
            video_path = "uploaded_video.mp4"
            with open(video_path, "wb") as f:
                f.write(uploaded_video.getbuffer())
            logger.info(f"Видео файл сохранен как {video_path}")
            st.write(f"Видео файл сохранен как {video_path}")

            st.video(video_path)

            with st.spinner('Анализ видео...'):
                # Анализ видео
                stats = analyze_video(video_path)
            st.header("Статистика по кандидату:")
            st.write(f"**Количество улыбок:** {stats.get('smiles', 0)}")
            st.write(f"**Количество морганий:** {stats.get('blinks', 0)}")
            st.write(f"**Количество признаков стресса:** {stats.get('stress', 0)}")
            st.write(f"**Количество движений:** {stats.get('movement', 0)}")

            # Анализ зрительного контакта
            st.write(f"**Устойчивые зрительные контакты:** {stats['eye_contact'].get('steady_eye_contact', 0)}")
            st.write(f"**Частые отводы взгляда:** {stats['eye_contact'].get('frequent_gaze_averts', 0)}")

            # Анализ эмоций
            st.subheader("Распознанные эмоции:")
            if stats.get('emotions') and len(stats['emotions']) > 0:
                for emotion, count in stats['emotions'].items():
                    st.write(f"**{emotion.capitalize()}:** {count}")
            else:
                st.write("Эмоции не распознаны.")

            # Преобразование видео в аудио
            audio_path = "extracted_audio.wav"
            extract_audio(video_path, audio_path)
            logger.info(f"Аудио файл извлечен как {audio_path}")
            st.write(f"Аудио файл извлечен как {audio_path}")

            # Транскрибация аудио
            with st.spinner('Транскрибация аудио...'):
                transcription = transcribe_audio(audio_path)
            st.header("Транскрибация аудио:")
            st.write(transcription)

            # Анализ аудио
            with st.spinner('Анализ аудио...'):
                audio_stats = analyze_audio(audio_path)
            st.header("Анализ аудио:")
            st.write(f"**Тональность голоса:** {audio_stats['tonality']}")
            st.write(f"**Тембр голоса:** {audio_stats['timbre']}")
            st.write(f"**Скорость речи:** {audio_stats['speech_rate']}")
            st.write(f"**Частота пауз:** {audio_stats['pause_frequency']}")
            st.write(f"**Использование заполняющих звуков:** {audio_stats['filler_sounds']}")

            # Сохранение результатов анализа в текстовый файл
            analysis_results_path = "analysis_results.txt"
            with open(analysis_results_path, "w", encoding="utf-8") as f:
                f.write("Результаты анализа видео:\n")
                f.write(f"Количество улыбок: {stats.get('smiles', 0)}\n")
                f.write(f"Количество морганий: {stats.get('blinks', 0)}\n")
                f.write(f"Количество признаков стресса: {stats.get('stress', 0)}\n")
                f.write(f"Количество движений: {stats.get('movement', 0)}\n")
                f.write(f"Количество устойчивых зрительных контактов: {stats['eye_contact'].get('steady_eye_contact', 0)}\n")
                f.write(f"Количество частых отводов взгляда: {stats['eye_contact'].get('frequent_gaze_averts', 0)}\n")
                f.write("\nРаспознанные эмоции:\n")
                if stats.get('emotions') and len(stats['emotions']) > 0:
                    for emotion, count in stats['emotions'].items():
                        f.write(f"{emotion.capitalize()}: {count}\n")
                else:
                    f.write("Эмоции не распознаны.\n")
                f.write("\nТранскрибация аудио:\n")
                f.write(transcription + "\n")
                f.write("\nАнализ аудио:\n")
                f.write(f"Тональность голоса: {audio_stats['tonality']}\n")
                f.write(f"Тембр голоса: {audio_stats['timbre']}\n")
                f.write(f"Скорость речи: {audio_stats['speech_rate']}\n")
                f.write(f"Частота пауз: {audio_stats['pause_frequency']}\n")
                f.write(f"Использование заполняющих звуков: {audio_stats['filler_sounds']}\n")
            logger.info(f"Результаты анализа сохранены в {analysis_results_path}")
            st.success(f"✅ Анализ завершён и результаты сохранены в {analysis_results_path}")

            # Отправка результатов анализа в GigaChat
            with st.spinner('Генерация отчета...'):
                try:
                    # Чтение содержимого файла
                    with open(analysis_results_path, "r", encoding="utf-8") as f:
                        file_content = f.read()
                        logger.info(f"Содержимое файла analysis_results.txt:\n{file_content}")
                    
                    # Получение токена GigaChat
                    token = get_access_token()
                    if not token:
                        st.error("Не удалось получить токен GigaChat.")
                        logger.error("Не удалось получить токен GigaChat.")
                    else:
                        # Генерация детализированного отчета
                        report = generate_analysis_report(file_content, token)
                        if report:
                            st.header("Детализированный отчет по кандидату:")
                            st.write(report)
                            logger.info("Отчет успешно отображен.")
                        else:
                            st.error("Не удалось сгенерировать отчет.")
                            logger.error("Не удалось сгенерировать отчет.")
                except Exception as e:
                    logger.error(f"Ошибка при обработке файла или запросе к GigaChat: {e}")
                    st.error(f"Ошибка: {e}")

            # Удаление временных файлов
            try:
                os.remove(video_path)
                os.remove(audio_path)
                os.remove(analysis_results_path)
                logger.info("Временные файлы удалены.")
                st.write("Временные файлы удалены.")
            except Exception as e:
                logger.warning(f"Не удалось удалить временные файлы: {e}")
                st.warning(f"Не удалось удалить временные файлы: {e}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке и обработке видео: {e}")
            st.error(f"Ошибка при загрузке и обработке видео: {e}")

# Вкладка 2: Анализ Текстовых и PDF Файлов
with tabs[1]:
    st.header("Анализ текстового или PDF файла для определения типа личности")

    uploaded_file = st.file_uploader("Загрузите текстовый (.txt) или PDF файл (.pdf)", type=["txt", "pdf"])

    if uploaded_file is not None:
        file_type = uploaded_file.type
        if uploaded_file.type == "application/pdf":
            st.write("Загружен PDF файл.")
            file_extension = "pdf"
        elif uploaded_file.type == "text/plain":
            st.write("Загружен текстовый файл (.txt).")
            file_extension = "txt"
        else:
            st.error("Поддерживаются только файлы с расширениями .txt и .pdf")
            st.stop()

        # Сохранение загруженного файла во временную директорию
        filename = secure_filename(uploaded_file.name)
        temp_dir = "uploads"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, filename)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"Файл '{filename}' успешно загружен.")

        with st.spinner('Извлечение текста из файла...'):
            # Обработка файла и извлечение текста
            result = process_file(uploaded_file, file_extension)

        if "error" in result:
            st.error(result["error"])
        else:
            extracted_text = result["text"]
            st.subheader("Извлеченный текст:")
            preview_length = 500
            if len(extracted_text) > preview_length:
                st.write(extracted_text[:preview_length] + "...")
            else:
                st.write(extracted_text)

            with st.spinner('Анализ текста и генерация отчетов...'):
                # Получение токена GigaChat
                token = get_access_token()
                if not token:
                    st.error("Не удалось получить токен GigaChat.")
                else:
                    # Генерация детализированного отчета и оценки резюме
                    report = generate_analysis_report(extracted_text, token)
                    evaluation = generate_resume_evaluation(extracted_text, token)

                    if report and evaluation:
                        st.header("Детализированный отчет по кандидату:")
                        st.write(report)
                        st.header("Оценка резюме кандидата:")
                        st.write(evaluation)
                    else:
                        st.error("Не удалось сгенерировать отчеты.")

        # Удаление временного файла
        try:
            os.remove(file_path)
            logger.info(f"Временный файл {file_path} удален.")
        except Exception as e:
            logger.warning(f"Не удалось удалить временный файл {file_path}: {e}")
            st.warning(f"Не удалось удалить временный файл {file_path}: {e}")
