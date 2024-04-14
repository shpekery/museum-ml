import os
import shutil

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import cv2
from starlette.responses import FileResponse
from transformers import AutoProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration, AutoModelForCausalLM

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io
from googletrans import Translator
from ultralytics import YOLO
import uvicorn
from heapq import nlargest


from PIL import Image

from image_similarity.data_preprocesing import get_idx, create_dataset
from image_similarity.func import get_embeddings, fetch_similar

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name_sim = "openai/clip-vit-large-patch14"
model_name_cap_git = "microsoft/git-base-coco"
model_name_cap_blip = "Salesforce/blip-image-captioning-large"

model_sim = CLIPModel.from_pretrained(model_name_sim).to(device)
print("CLIPModel loaded...")
processor_sim = AutoProcessor.from_pretrained(model_name_sim)
print("AutoProcessor loaded...")

model_cap_git = AutoModelForCausalLM.from_pretrained(model_name_cap_git).to(device)
print("Caption Git loaded...")
processor_cap_git = AutoProcessor.from_pretrained(model_name_cap_git)
print("AutoProcessorGit loaded...")

model_cap_blip = BlipForConditionalGeneration.from_pretrained(model_name_cap_blip).to(device)
print("Caption BLIP loaded...")
processor_cap_blip = BlipProcessor.from_pretrained(model_name_cap_blip)
print("BLIP Processor loaded...")

seed = 42
batch_size = 64

# Load a pretrained YOLOV8 model

yolo_model = YOLO(
    r"classification/best.pt")
print("YOLO Classifier loaded...")

translator = Translator()


def translate_text(text, dest_lang='ru', src='en'):
    while True:
        try:
            translation = translator.translate(text, dest=dest_lang, src=src)
            return translation.text
        except:
            continue


def generate_caption_git(image):
    pixel_values = processor_cap_git(images=image, text='a museum photography of', return_tensors="pt").pixel_values.to(
        device)

    generated_ids = model_cap_git.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor_cap_git.batch_decode(generated_ids, skip_special_tokens=True)[0].replace("a museum photography of", " ")
    return translate_text(generated_caption)


def generate_caption_blip(image):
    # conditional image captioning
    text = "a museum photography of"
    inputs = processor_cap_blip(image, text, return_tensors="pt").to(device)

    out = model_cap_blip.generate(**inputs)
    answer = processor_cap_blip.decode(out[0], skip_special_tokens=True).replace("a museum photography of", " ")
    return translate_text(answer)


def get_transformations():
    # Data transformation chain.
    return T.Compose(
        [
            # We first resize the input image to 256x256 and then we take center crop.
            T.Resize((256, 256)),
            T.CenterCrop((192, 192)),
        ]
    )


def get_subset_embeddings(data):
    def extract_embeddings(batch):
        """Utility to compute embeddings."""
        return {"embeddings": batch["image"]}

    candidate_subset_emb = data.map(extract_embeddings, batched=True, batch_size=batch_size)
    all_candidate_embeddings = torch.from_numpy(np.array(candidate_subset_emb["embeddings"]))
    candidate_ids = get_idx(candidate_subset_emb)

    return all_candidate_embeddings, candidate_ids


def create_image_embedding(img, group):
    embeddings_model = get_embeddings(processor=processor_sim,
                                      model=model_sim,
                                      image=img)
    return torch.from_numpy(np.insert(np.array(embeddings_model.data.cpu()).squeeze(0), 0, group))


def get_image_group(img, n_group):
    pred = yolo_model.predict(img)

    def find_n_max(n, data, pred):
        proba = [nlargest(n, x)[n - 1] for x in data][0]
        i = [x.index(nlargest(n, x)[n - 1]) for x in data][0]
        predicted_max = pred[0].names[i]
        return [predicted_max, i, proba]

    groups = []

    a = [r.probs.data for r in pred]
    b = [c.cpu().numpy().tolist() for c in a]

    for i in range(1, n_group + 1):
        groups.append(find_n_max(i, b, pred))

    return groups


def find_valid_scores(scores):
    threshold = 0.85
    return [x >= threshold for x in scores]


def predict_top_k(img, group, all_embeddings, candidate_ids, k=10):
    # sample = Image.open(img)  # test_img[img] if isinstance(img, int) else

    sample = [get_transformations()(img)]
    with torch.no_grad():
        embeddings = create_image_embedding(sample, group)
        # embeddings_2 = get_embeddings(processor=processor_2,
        #                               model=model_2,
        #                               image=sample)

        # emb = torch.hstack((embeddings_1, embeddings_2))

        sim_idx, sim_labels, sim_paths, sim_scores = fetch_similar(query_embeddings=embeddings,
                                                                   all_embeddings=all_embeddings,
                                                                   idx=candidate_ids, top_k=k)

    # orig_path = cv2.imread(img) # test['image_file_path'][img] if isinstance(img, int) else

    images = [cv2.imread(x) for x in sim_paths]

    for i in range(len(images)):
        images[i] = cv2.resize(images[i], (255, 255))

    # concatenate image Vertically
    vert = np.concatenate(images, axis=0)

    # cv2.imwrite("OrigImg.jpg", orig_path)
    cv2.imwrite('TopK.jpg', vert)

    valid_scores = find_valid_scores(sim_scores)

    return sim_labels, sim_scores, valid_scores


def get_img_description(image):
    return {"caption_1": generate_caption_blip(image),
            "caption_2": generate_caption_git(image)}


def load():
    dataset, paths = create_dataset("train_emb.csv", train_test_percentage=0)
    print("Dataset loaded...")
    all_candidate_embeddings, candidate_ids = get_subset_embeddings(dataset)
    return all_candidate_embeddings, candidate_ids, paths


all_candidate_embeddings, candidate_ids, paths = load()
descriptions = pd.read_csv("train.csv", sep=';')

# transform_to_embeddings_dataset("E:\MachineLearningProjects\ml-practices\src\haccaton\data//train", "data/train.csv", get_transformations(), processor=processor_sim, model=model_sim)
#
# Fast API Code
app = FastAPI(title="Similarity Finder Tool", description="An API for finding similar images with an input pic.")

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SimilarityModel(BaseModel):
    caption: str


# Redirect to user docs
@app.get("/", include_in_schema=False)
def index():
    return RedirectResponse(url="/docs")


@app.post("/predict/", response_model=SimilarityModel)
def predict(file: UploadFile = File(...), description: bool = True):
    # Load the image file
    content = file.file.read()
    image = Image.open(io.BytesIO(content))
    img_description = None

    if description:
        img_description = get_img_description(image)

    print("Start image group...")
    groups = get_image_group(image, 3)

    max_group = groups[0][1]

    groups = list(map(lambda x: [x[0], x[2]], groups))

    print("Finding similar images...")
    sim_labels, sim_scores, valid_scores = predict_top_k(image, max_group, all_candidate_embeddings, candidate_ids,
                                                         k=10)

    sim_scores = list(map(str, sim_scores))

    tmp = [(x.split(".")[0], ".".join(x.split(".")[1:])) for x in sim_labels]
    _name = []
    _description = []
    _class = []

    for _id, img_name in tmp:
        print(_id, img_name)
        _n, _d, _c = (descriptions[(descriptions["object_id"] == int(_id)) & (descriptions["img_name"] == img_name)][
            ["name", "description", "group"]]).values.tolist()[0]
        _name += [_n]
        _description += [_d if str(_d) != 'nan' else "None"]
        _class += [str(_c)]

    ans = list(zip(sim_labels, _name, _description, _class, sim_scores, valid_scores))
    res = {"class": groups,
           "similar_images": ans,
           }

    if description:
        res['description'] = img_description

    return JSONResponse(content=res,
                        status_code=200)


@app.post("/gen_description/", response_model=SimilarityModel)
def gen_description(file: UploadFile = File(...)):
    # Load the image file
    content = file.file.read()
    image = Image.open(io.BytesIO(content))

    img_description = get_img_description(image)
    return JSONResponse(content={
                                 "description": img_description,
                                 },
                        status_code=200)

@app.get("/images/")
async def main(fold, img):
    idx = f"{fold}.{img}"
    _path = list(paths[paths["object_id"].str.contains(idx)]["img_path"])[0]
    return FileResponse(_path)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
