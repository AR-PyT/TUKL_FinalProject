from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import cv2
# import time
import threading

mobile_net_v2_1 = tf.keras.models.load_model("includes/mobilenetv2_98.h5")
mobile_net_v2_2 = tf.keras.models.load_model("includes/mobilenetv2_98_2.h5")
mobile_net_v2_3 = tf.keras.models.load_model("includes/mobilenetv2_97_7.h5")

augmenter = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomZoom((-0.2, 0.4)),
    tf.keras.layers.experimental.preprocessing.CenterCrop(120, 120),
    tf.keras.layers.experimental.preprocessing.Resizing(160, 160)
])


def make_prediction(models, images, batch):
    results = []

    def get_predict(model, image):
        results.append(np.count_nonzero(tf.math.argmax(tf.nn.softmax(model(image)), axis=1).numpy()) > 2)

    threads = []
    for model, index in zip(models, range(1, len(models) + 1)):
        threads.append(threading.Thread(target=lambda: get_predict(model, images[(index - 1) * batch:index * batch])))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    dogs = len([x for x in results if not x])
    cats = len([x for x in results if x])

    return cats > dogs


class Image(BaseModel):
    image: bytes


app = FastAPI()


@app.post("/")
async def root(image: UploadFile = File(...)):
    try:
        image = image.file.read()
        img = cv2.imdecode(np.frombuffer(image, np.uint8), -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dsize=(160, 160))
        img = np.asarray(img)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error in provided Image: " + str(e))
    imgs = augmenter(np.array([img] * 9)).numpy()
    # s = time.time()
    output = make_prediction(
        [mobile_net_v2_1, mobile_net_v2_2, mobile_net_v2_3],
        imgs,
        3
    )
    # e = time.time()
    # print(output)
    # print(e - s)
    return {"prediction": output}
