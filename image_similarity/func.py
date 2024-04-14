import torch.nn as nn


def get_embeddings(processor, model, image):
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    image_features = model.get_image_features(**inputs).to(model.device)
    return image_features


def compute_scores(emb_one, emb_two):
    """Computes cosine similarity between two vectors."""
    emb_one = emb_one.to("cpu")
    emb_two = emb_two.to("cpu")
    scores = nn.functional.cosine_similarity(emb_one, emb_two)
    # print(scores.data.numpy())
    return scores.data.numpy().tolist()


def fetch_similar(query_embeddings, all_embeddings, idx, top_k=5):
    """Fetches the `top_k` similar images with `image` as the query."""
    # Prepare the input query image for embedding computation.
    # image_transformed = transformation_chain(image).unsqueeze(0)

    # Compute similarity scores with all the candidate images at one go.
    # We also create a mapping between the candidate image identifiers
    # and their similarity scores with the query image.
    sim_scores = compute_scores(all_embeddings, query_embeddings)
    similarity_mapping = dict(zip(idx, sim_scores))

    # Sort the mapping dictionary and return `top_k` candidates.
    similarity_mapping_sorted = dict(
        sorted(similarity_mapping.items(), key=lambda x: x[1], reverse=True)
    )
    id_entries = list(similarity_mapping_sorted.keys())[:top_k]
    sim_scores = list(similarity_mapping_sorted.values())[:top_k]

    ids = list(map(lambda x: int(x.split("|")[0]), id_entries))
    labels = list(map(lambda x: x.split("|")[1], id_entries))
    paths = list(map(lambda x: x.split("|")[2], id_entries))
    return ids, labels, paths, sim_scores