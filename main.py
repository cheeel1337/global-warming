!wget https://github.com/OlafenwaMoses/ImageAI/releases/download/3.0.0-pretrained/yolov3.pt
from imageai.Detection import ObjectDetection
import os
import cv2
from PIL import Image
from imageai.Detection import VideoObjectDetection

def detect_video(model_path='yolov3.pt', video_path='video.mp4'):
    execution_path = os.getcwd()
    video_full_path = os.path.join(execution_path, video_path)

    if not os.path.exists(video_full_path):
        print(f'{video_full_path}')
        return None

    video = cv2.VideoCapture(video_full_path)
    if not video.isOpened():
        print(f'{video_full_path}')
        return None

    detector = VideoObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(os.path.join(execution_path, model_path))

    try:
        detector.loadModel()
    except Exception as e:
        print(f'{e}')
        video.release()
        return None

    file_name_without_extension = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(execution_path, f'output_{file_name_without_extension}.mp4')

    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        returned_video_path, count = detector.detectObjectsFromVideo(
            input_file_path=video_full_path,
            output_file_path=output_path,
            frames_per_second=20,
            log_progress=True
        )
    except Exception as e:
        print(f'{e}')
        video.release()
        return None

    video.release()

    return returned_video_path

if __name__ == '__main__':
    result_detection = detect_video(model_path='yolov3.pt', video_path='walking.mp4')
    if result_detection:
        print(f'выходное видео: {result_detection}')
    else:
        print('ошибка обработки.')

def detect_objects_on_road(image, model = 'yolov3.pt', percentage = 30):
    detector = ObjectDetection()
    model_path = model
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(model_path)
    detector.loadModel()

    detections = detector.detectObjectsFromImage(
        input_image=image,
        output_image_path=f"output_{image}",
        minimum_percentage_probability=percentage)
    return detections











