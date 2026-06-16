from base64 import b64decode
from io import BytesIO
from contextlib import asynccontextmanager

import numpy as np
from pandas import read_csv
from onnxruntime import InferenceSession
from PIL import Image, ImageFile

import uvicorn
from fastapi import FastAPI, Request, HTTPException

ImageFile.LOAD_TRUNCATED_IMAGES = True

kaomojis = [
    "0_0",
    "(o)_(o)",
    "+_+",
    "+_-",
    "._.",
    "<o>_<o>",
    "<|>_<|>",
    "=_=",
    ">_<",
    "3_3",
    "6_9",
    ">_o",
    "@_@",
    "^_^",
    "o_o",
    "u_u",
    "x_x",
    "|_|",
    "||_||"
]

def load_labels(df):
    tag_names = [
        n.replace("_", " ") if n not in kaomojis else n
        for n in df["name"]
    ]

    # rating_idx = list(np.where(df["category"] == 9)[0])
    general_idx = list(np.where(df["category"] == 0)[0])
    character_idx = list(np.where(df["category"] == 4)[0])

    return tag_names, general_idx, character_idx

def mcut_threshold(probs):
    """
    Maximum Cut Thresholding (MCut)
    Largeron, C., Moulin, C., & Gery, M. (2012). MCut: A Thresholding Strategy
     for Multi-label Classification. In 11th International Symposium, IDA 2012
     (pp. 172-183).
    """
    sorted_probs = probs[probs.argsort()[::-1]]
    difs = sorted_probs[:-1] - sorted_probs[1:]
    t = difs.argmax()
    thresh = (sorted_probs[t] + sorted_probs[t + 1]) / 2
    return thresh

class Predictor:
    def __init__(self):
        self.model = None
        self.model_target_size = None
        self.input_name = None
        self.label_name = None

        self._load_model()

    def _load_model(self):
        tags_df = read_csv("selected_tags.csv")
        self.tag_names, self.general_idx, self.character_idx = load_labels(tags_df)

        model = InferenceSession("model.onnx", providers=["CUDAExecutionProvider"])
        self.model_target_size = model.get_inputs()[0].shape[1]
        self.input_name = model.get_inputs()[0].name
        self.label_name = model.get_outputs()[0].name
        self.model = model

    def _prepare_image(self, image):
        image = Image.alpha_composite(
            Image.new("RGBA", image.size, (255, 255, 255, 255)),
            image.convert("RGBA") if image.mode != "RGBA" else image
        ).convert("RGB")

        w, h = image.size
        max_dim = max(w, h)
        square = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        square.paste(image, ((max_dim - w) // 2, (max_dim - h) // 2))

        target_size = self.model_target_size
        square = square.resize((target_size, target_size), Image.BICUBIC) if max_dim != target_size else square

        image_array = np.asarray(square, dtype=np.float32)
        image_array = image_array[:, :, ::-1] # Convert PIL-native RGB to BGR

        return np.expand_dims(image_array, axis=0)

    def predict(
        self,
        image,
        general_thresh,
        general_mcut_enabled,
        character_thresh,
        character_mcut_enabled,
    ):
        image = self._prepare_image(image)
        preds = self.model.run([self.label_name], {self.input_name: image})[0]
        labels = list(zip(self.tag_names, preds[0].astype(float)))

        # rating = {tag: score for tag, score in (labels[i] for i in self.rating_idx)}

        def _apply_threshold(idx, mcut_enabled, base_thresh, min_thresh=None):
            tag_scores = [labels[i] for i in idx]
    
            if mcut_enabled:
                probs = np.fromiter((score for _, score in tag_scores), dtype=np.float32)
                base_thresh = mcut_threshold(probs)
                base_thresh = max(min_thresh, base_thresh) if min_thresh is not None else base_thresh

            return {tag: score for tag, score in tag_scores if score > base_thresh}

        general_res = _apply_threshold(self.general_idx, general_mcut_enabled, general_thresh)
        character_res = _apply_threshold(self.character_idx, character_mcut_enabled, character_thresh, min_thresh=0.15)
        return general_res, character_res

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = Predictor()
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/tagger/v1/interrogate")
async def interrogate(request: Request):
    body = await request.json()

    encoding = body["image"]
    encoding = encoding.split(",")[1] if encoding.startswith("data:") else encoding
    image = Image.open(BytesIO(b64decode(encoding)))

    general_res, character_res = request.app.state.predictor.predict(
        image,
        body.get("general_thresh", 0.35),
        body.get("general_mcut_enabled", False),
        body.get("character_thresh", 0.85),
        body.get("character_mcut_enabled", False)
    )

    return { "caption": { "tag": { **general_res, **character_res } } }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7861)