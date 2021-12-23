from numpy.core.numeric import outer
from tensorflow.python.keras.backend import dtype
from app import app
from app import cloud
from flask import request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from mtcnn.mtcnn import MTCNN
import cv2
import urllib.request
import numpy as np
import cloudinary.uploader
import os
# from app.models.recomendations import db, Recomendations

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# with open('model/faceshape_model.json', 'r') as json_file:
#     json_savedModel = json_file.read()
# model = tf.keras.models.model_from_json(json_savedModel)
model = load_model(
    'model/modelgender.h5')


def cropAndResize(image, targetW=128, targetH=128):
    if image.ndim == 2:
        imgH, imgW = image.shape
    elif image.ndim == 3:
        imgH, imgW, channels = image.shape
    targetAspectRatio = targetW/targetH
    inputAspectRatio = imgW/imgH

    if inputAspectRatio > targetAspectRatio:
        resizeW = int(inputAspectRatio*targetH)
        resizeH = targetH
        img = cv2.resize(image, (resizeW, resizeH))
        cropLeft = int((resizeW - resizeH)/2)
        cropRight = cropLeft + targetW
        newImage = img[:, cropLeft:cropRight]
    if inputAspectRatio < targetAspectRatio:
        resizeW = targetW
        resizeH = int(inputAspectRatio/targetW)
        img = cv2.resize(image, (resizeW, resizeH))
        cropTop = int((resizeH - targetH) / 4)
        cropBottom = cropTop + targetH
        newImage = img[cropTop:cropBottom, :]
    if inputAspectRatio == targetAspectRatio:
        newImage = cv2.resize(image, (targetW, targetH))
    return newImage


def extract_face(image, target_size=(128, 128)):
    detector = MTCNN()
    result = detector.detect_faces(image)
    if result == []:
        newFace = cropAndResize(image, targetW=128, targetH=128)
    else:
        x1, y1, width, height = result[0]['box']
        x2, y2 = x1+width, y1+height
        face = image[y1:y2, x1:x2]
        adjH = 10
        if y1-adjH < 10:
            newY1 = 0
        else:
            newY1 = y1 - adjH
        if y1+height+adjH < image.shape[0]:
            newY2 = y1+height+adjH
        else:
            newY2 = image.shape[0]
        newHeight = newY2 - newY1
        adjW = int((newHeight-width)/2)
        if x1-adjW < 0:
            newX1 = 0
        else:
            newX1 = x1 - adjW
        if x2+adjW > image.shape[1]:
            newX2 = image.shape[1]
        else:
            newX2 = x2+adjW
            newFace = image[newY1:newY2, newX1:newX2]
        squareImage = cv2.resize(newFace, target_size)
        return squareImage


def predictGender(imageArray):
    # labelGender = {0: 'female', 1: 'male'}
    try:
        faceImage = extract_face(imageArray)
        newImage = cv2.cvtColor(faceImage, cv2.COLOR_BGR2RGB)
        testImage = np.array(newImage, dtype=float)
        testImage = testImage/255
        testImage = np.array(testImage).reshape(1, 128, 128, 3)

        pred = model.predict(testImage)[0][0]
        print(pred)
        if pred <= 0.6:
            return 'female', (1-pred)
        else:
            return 'male', (pred)
    except Exception as e:
        return({
            'error': "Oops! Something went wrong. Please try again"
        })


def result():
    if 'image' not in request.files:
        resp = jsonify({'msg': "No body image attached in request"})
        resp.status_code = 501
        return resp
    image = request.files['image']
    if image.filename == '':
        resp = jsonify({'msg': "No file image selected"})
        resp.status_code = 404
        return resp
    error = {}
    success = False

    if image and allowed_file(image.filename):
        upload_result = cloudinary.uploader.upload(image)
        print(upload_result["secure_url"])
        success = True
    else:
        error[image.filename] = "File type is not allowed"

    if success and error:
        error['Message'] = "File not uploaded"
        resp = jsonify(error)
        resp.status_code = 500
        return resp
    if success:
        try:
            readImageFromUrl = urllib.request.urlopen(
                upload_result["secure_url"])
            image = np.asarray(
                bytearray(readImageFromUrl.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            output, prob = predictGender(image)
            result_prob = str(prob)
            return jsonify({
                'status': 200,
                'msg': "Success get predict gender",
                'Gender': output,
                'Probability': result_prob
            })
        except Exception as e:
            resp = {

                'status': 500,
                'msg': "Failed get predict gender",
                'Error': "Image yang masukan bukan wajah"

            }
            error = jsonify(resp)
            error.status_code = 500
            return error
