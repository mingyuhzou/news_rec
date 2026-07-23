import os
import re
import zipfile

import numpy as np
import pandas as pd
import polars as pl

from tqdm import tqdm
from os.path import join
from sentence_transformers import SentenceTransformer

from config.common import cfg


# =========================
# Process impression
# =========================

def process_impression(row):

    history = row["history"]

    if pd.isna(history):
        history_items = []
    else:
        history_items = str(history).split()


    click_items = []
    unclick_items = []


    impressions = row["impressions"]

    if not pd.isna(impressions):

        impressions = str(impressions).strip("[]")

        for item in re.split(r"[\s.]+", impressions):

            if not item:
                continue

            parts = item.rsplit("-", 1)

            if len(parts) != 2:
                continue

            news_id, label = parts


            if label == "1":
                click_items.append(news_id)

            elif label == "0":
                unclick_items.append(news_id)


    # click加入history
    history_items.extend(click_items)


    return (
        " ".join(history_items),
        click_items,
        unclick_items
    )



# =========================
# Save data
# =========================

def save_processed_data(
        train_df,
        test_df
):

    os.makedirs(
        cfg["train_data_path"],
        exist_ok=True
    )

    os.makedirs(
        cfg["test_data_path"],
        exist_ok=True
    )


    train_df.write_parquet(
        join(
            cfg["train_data_path"],
            "train.parquet"
        )
    )


    test_df.write_parquet(
        join(
            cfg["test_data_path"],
            "test.parquet"
        )
    )



def prepare_data(data_list):

    rows=[]

    for item in data_list:

        rows.append(
            {
                "user_id": item["user_id"],
                "history": item["history"],
                "target": item["target"],
                "click": item["click"],
                "unclick": item["unclick"]
            }
        )


    return pl.DataFrame(rows)


# =========================
# Main preprocess
# =========================

def preprocessData():

    print("="*50)
    print("Start preprocessing")
    print("="*50)


    data_path = cfg["data_path"]


    # =========================
    # unzip
    # =========================

    print("[1/8] Checking zip file...")


    zip_path = join(
        data_path,
        cfg['file_name'],
        'zip'
    )


    extract_path = join(
        data_path,
        cfg['file_name']
    )


    if not os.path.exists(extract_path):

        print("Extracting dataset...")

        with zipfile.ZipFile(
            zip_path,
            "r"
        ) as zip_ref:

            zip_ref.extractall(
                data_path
            )

        print("Extraction finished")

    else:
        print("Dataset already extracted")


    data_path = extract_path



    # =========================
    # behavior
    # =========================

    print("[2/8] Loading behaviors.tsv...")


    behaviors = pd.read_csv(
        join(
            data_path,
            "behaviors.tsv"
        ),
        sep="\t",
        header=None,
        names=[
            "index",
            "user_id",
            "timestamp",
            "history",
            "impressions"
        ]
    )


    print(
        f"Loaded behaviors: {len(behaviors)}"
    )


    behaviors["timestamp"] = pd.to_datetime(
        behaviors["timestamp"],
        format="%m/%d/%Y %I:%M:%S %p",
        errors="coerce"
    )



    # =========================
    # impression
    # =========================

    print("[3/8] Processing impressions...")


    processed = behaviors.apply(
        process_impression,
        axis=1
    )


    behaviors["history"] = processed.apply(
        lambda x:x[0]
    )

    behaviors["click"] = processed.apply(
        lambda x:x[1]
    )

    behaviors["unclick"] = processed.apply(
        lambda x:x[2]
    )


    print("Impression processing finished")



    # =========================
    # user mapping
    # =========================

    print("[4/8] Building user mapping...")


    user_ids = (
        behaviors["user_id"]
        .unique()
        .tolist()
    )


    print(
        f"Number of users: {len(user_ids)}"
    )


    user2idx = {
        u:i+1
        for i,u in enumerate(user_ids)
    }


    behaviors["user_id"] = (
        behaviors["user_id"]
        .map(user2idx)
    )


    np.save(
        cfg["user_dict"],
        user2idx
    )


    print("User mapping saved")



    # =========================
    # news mapping
    # =========================

    print("[5/8] Loading news.tsv...")


    news = pd.read_csv(
        join(
            data_path,
            "news.tsv"
        ),
        sep="\t",
        header=None,
        names=[
            "news_id",
            "category",
            "subcategory",
            "title",
            "abstract",
            "url",
            "title_entities",
            "abstract_entities"
        ]
    )


    print(
        f"Number of news: {len(news)}"
    )


    news_ids = (
        news["news_id"]
        .astype(str)
        .tolist()
    )


    history_ids = (
        behaviors["history"]
        .astype(str)
        .str.split()
        .explode()
        .unique()
        .tolist()
    )


    all_news = list(
        dict.fromkeys(
            news_ids + history_ids
        )
    )


    print(
        f"Total items: {len(all_news)}"
    )


    news2idx = {
        n:i+1
        for i,n in enumerate(all_news)
    }


    np.save(
        cfg["item_dict"],
        news2idx
    )


    print("Item mapping saved")



    # =========================
    # mapping
    # =========================

    print("[6/8] Mapping history/click/unclick...")


    def map_history(history):

        result=[]

        for item in history.split():

            if item in news2idx:

                result.append(
                    str(news2idx[item])
                )

        return " ".join(result)



    def map_list(items):

        return [
            news2idx[x]
            for x in items
            if x in news2idx
        ]



    behaviors["history"] = (
        behaviors["history"]
        .apply(map_history)
    )
    behaviors["click"] = (
        behaviors["click"]
        .apply(map_list)
    )
    behaviors["unclick"] = (
        behaviors["unclick"]
        .apply(map_list)
    )


    print("Mapping finished")



    # =========================
    # train test
    # =========================

    print("[7/8] Preparing train/test...")


    train={}
    test={}

    train_click = {}
    train_unclick = {}

    test_click = {}
    test_unclick = {}

    for _,row in behaviors.iterrows():

        seq = row["history"].split()


        if len(seq)>2:

            train[row["user_id"]] = seq[:-1]

            test[row["user_id"]] = seq

            train_click[row["user_id"]] = row["click"]
            train_unclick[row["user_id"]] = row["unclick"]

            test_click[row["user_id"]] = row["click"]
            test_unclick[row["user_id"]] = row["unclick"]



    print(f"Train users: {len(train)}")
    print(f"Test users: {len(test)}")

    train_df = prepare_data(
        train,
        train_click,
        train_unclick
    )
    test_df = prepare_data(
        test,
        test_click,
        test_unclick
    )


    save_processed_data(
        train_df,
        test_df
    )

    print("Train/test saved")

    # =========================
    # item embedding
    # =========================

    print("[8/8] Generating item embeddings...")
    model = SentenceTransformer(
        cfg["emb_model_path"],
        device="cuda"
    )
    print("Embedding model loaded")

    item_embeddings=[]

    for _,row in tqdm(
        news.iterrows(),
        total=len(news),
        desc="Encoding news"
    ):

        news_id = row["news_id"]
        if news_id not in news2idx:
            continue

        item_id = news2idx[news_id]
        embedding = model.encode(
            row["title"],
            normalize_embeddings=True
        )
        item_embeddings.append(
            {
                "item_id":item_id,
                "embedding":embedding.tolist()
            }
        )

    print(f"Generated embeddings: {len(item_embeddings)}")


    item_emb_df = pd.DataFrame(item_embeddings)


    os.makedirs(cfg["embed_path"],exist_ok=True)


    item_emb_df.to_parquet(
        join(
            cfg["embed_path"],
            "item_emb_title.parquet"
        ),
        index=False
    )


    print("Embedding saved")

    print("="*50)
    print("Preprocessing finished")
    print("="*50)

    return item_emb_df


if __name__ == "__main__":
    preprocessData()