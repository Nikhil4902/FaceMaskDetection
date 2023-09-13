import imutils
from imutils.video import VideoStream
import numpy as np
import argparse
import time
import cv2
import tensorflow as tf
img_to_array = tf.keras.preprocessing.image.img_to_array
preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
load_model = tf.keras.models.load_model

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--prototxt', default='FaceDetectorModel/deploy.prototxt',
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument('-m', '--model', default='FaceDetectorModel/res10_300x300_ssd_iter_140000.caffemodel',
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")

args = vars(ap.parse_args())


def detect_face_and_predict_mask(frame, faceNet, maskNet):

	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	faceNet.setInput(blob)
	detections = faceNet.forward()


	faces = []
	locs = []
	preds = []

	for i in range(0, detections.shape[2]):

		confidence = detections[0, 0, i, 2]

		if confidence > args["confidence"]:

			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			face = frame[startY:endY, startX:endX]
			if face.any():
				face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
				face = cv2.resize(face, (224, 224))
				face = img_to_array(face)
				face = preprocess_input(face)

				faces.append(face)
				locs.append((startX, startY, endX, endY))

	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=16, verbose=0)

	return locs, preds


def main() -> None:
    print("[INFO] loading models...")
    face_detection_net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
    mask_detection_net = load_model('MaskDetectorModel/mask_detector_mobilenet_v2.model')
    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    while True:

        frame = vs.read()
        frame = imutils.resize(frame, width=1300)
        frame = cv2.flip(frame, 1) # type: ignore

        locs, preds = detect_face_and_predict_mask(frame, face_detection_net, mask_detection_net)

        for box, pred in zip(locs, preds):

            startX, startY, endX, endY = box
            mask, withoutMask = pred

            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            cv2.putText(frame, label, (startX, startY - 10), # type: ignore
                cv2.FONT_HERSHEY_PLAIN, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2) # type: ignore

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()


if __name__ == '__main__':
    main()
