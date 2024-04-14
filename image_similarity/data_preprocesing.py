import numpy as np
import os

import torch
from datasets import Dataset
from PIL import Image
import pandas as pd

from image_similarity.func import get_embeddings


def make_single_dataset(x, y):
    res = {'image_file_path': list(y['img_path']),
           'image': [torch.tensor(list(map(lambda k: np.float16(k), elem))) for elem in x.values],
           'labels': y['object_id']}

    return Dataset.from_dict(res)


def create_dataset(csv_path, train_test_percentage=0.2):
    df = pd.read_csv(csv_path)

    from sklearn.model_selection import train_test_split
    conv = {x: i for i, x in enumerate(df['group'].unique())}
    df['group'] = df['group'].apply(lambda x: conv[x])
    X = df.drop(["Unnamed: 0", "object_id", "img_path"], axis=1)
    Y = df[["object_id", "img_path"]]

    if train_test_percentage > 0:
        train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=train_test_percentage)

        train_dataset = make_single_dataset(train_X, train_Y)
        test_dataset = make_single_dataset(test_X, test_Y)

        return train_dataset, test_dataset, train_Y, test_Y

    return make_single_dataset(X, Y), Y


def transform_to_embeddings_dataset(dir_path, data_path, transformation_chain, processor, model,
                                    save_name="train_emb.csv"):
    train_data = pd.read_csv(data_path, sep=';')[["object_id", "group", "img_name"]]

    ans = []
    cols = None
    c = 0
    for img_id, img_group, img_name in train_data.values:
        img_dir = os.path.join(dir_path, str(img_id), img_name)
        image = transformation_chain(Image.open(img_dir))

        with torch.no_grad():
            embeddings = get_embeddings(processor=processor,
                                        model=model,
                                        image=image)
            embeddings = embeddings.data.cpu().numpy().tolist()[0]

            if cols is None:
                features = [f"feature_{i}" for i in range(len(embeddings))]
                cols = ["object_id", "img_path", "group", *features]
                flag = False

            tmp = [f"{img_id}.{img_name}", img_dir, img_group, *embeddings]
            ans.append(tmp)

        if c % 100 == 0:
            print(c)
        c += 1
    print(c)
    res = pd.DataFrame(ans, columns=cols)
    res.to_csv(save_name)
    return res


def get_idx(data):
    from tqdm.auto import tqdm

    candidate_ids = []

    for id in tqdm(range(len(data))):
        label = data[id]["labels"]
        _path = data[id]["image_file_path"]

        # Create a unique indentifier.
        entry = str(id) + "|" + str(label) + "|" + str(_path)

        candidate_ids.append(entry)

    return candidate_ids


def get_transformations():
    import torchvision.transforms as T
    # Data transformation chain.
    return T.Compose(
        [
            # We first resize the input image to 256x256 and then we take center crop.
            T.Resize((256, 256)),
            T.CenterCrop((192, 192)),
        ]
    )