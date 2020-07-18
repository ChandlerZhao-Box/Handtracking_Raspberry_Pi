import argparse
import platform
import time

import cv2
import tflite_runtime.interpreter as tflite
from PIL import Image

import detect

EDGETPU_SHARED_LIB = {
    'Linux': 'libedgetpu.so.1',
    'Darwin': 'libedgetpu.1.dylib',
    'Windows': 'edgetpu.dll'
}[platform.system()]


def load_labels(path, encoding='utf-8'):
    with open(path, 'r', encoding=encoding) as f:
        lines = f.readlines()
    if not lines:
        return {}
    if lines[0].split(' ', maxsplit=1)[0].isdigit():
        pairs = [line.split(' ', maxsplit=1) for line in lines]
        return {int(index): label.strip() for index, label in pairs}
    else:
        return {index: line.strip() for index, line in enumerate(lines)}


def make_interpreter(model_file):
    model_file, *device = model_file.split('@')
    return tflite.Interpreter(model_path=model_file, experimental_delegates=[
        tflite.load_delegate(EDGETPU_SHARED_LIB, {'device': device[0]} if device else {})])


def draw_objects(draw, objs, labels):
    for obj in objs:
        bbox = obj.bbox
        draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)], outline='red')
        draw.text((bbox.xmin + 10, bbox.ymin + 10), '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score), fill='red')


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', default="models/hand_edgetpu.tflite",
                        help='File path of .tflite file.')
    parser.add_argument('-l', '--labels', default="models/labels.txt",
                        help='File path of labels file.')
    parser.add_argument('-t', '--threshold', type=float, default=0.4,
                        help='Score threshold for detected objects.')
    args = parser.parse_args()

    labels = load_labels(args.labels) if args.labels else {}
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()

    print('----INFERENCE TIME----')
    print('Note: The first inference is slow because it includes', 'loading the model into Edge TPU memory.')

    video_capture = cv2.VideoCapture(0)
    while video_capture.isOpened():
        start = time.time()
        ret, frame = video_capture.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        scale = detect.set_input(interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))
        interpreter.invoke()
        objs = detect.get_output(interpreter, args.threshold, scale)
        for obj in objs:
            print(obj)
            topLeftX = obj.bbox.xmin
            topLeftY = obj.bbox.ymin
            bottomRightX = obj.bbox.xmax
            bottomRightY = obj.bbox.ymax

            cv2.rectangle(frame, (topLeftX, topLeftY), (bottomRightX, bottomRightY), (0, 255, 0), 2)
            cv2.putText(frame, labels[obj.id], (topLeftX, topLeftY), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0), 1,
                        True)
        end = time.time()
        cv2.putText(frame, 'Press ESC to EXIT. FTPS: %2f' % (1 / (end - start)), (40, 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255),
                    0, True)
        cv2.imshow("Detection", frame)

        if cv2.waitKey(40) & 0xFF == 27:
            cv2.destroyAllWindows()
            break
    video_capture.release()
    cv2.destroyAllWindows()
    print("Finished")


if __name__ == '__main__':
    main()
