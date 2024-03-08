"""
This module contains helper functions used to reproduce results in *Defending Against
Authorship Attribution Attacks With Large Language Models*.
"""

__author__ = 'hw56@indiana.edu'
__license__ = 'OBSD'

import json
import os
import random
import re
from typing import List, Tuple

import chardet
import numpy as np
from sacremoses import MosesTokenizer


def get_corpus_stats() -> None:
    """
    Compute and display statistics about the number of tokens in different corpora and
    tasks.

    Supported corpora include 'rj', 'ebg', and 'lcmc', and each has its own set of
    samples. For each corpus and task, the function will output the average and standard
    deviation of the number of tokens in both training and testing data defined using
    the Moses tokenizer.
    """
    tokenize = lambda s: MosesTokenizer(lang="en").tokenize(s, escape=False)

    corpus_to_function = {"rj": load_rj, "ebg": load_ebg, "lcmc": load_lcmc}

    def _corpus_stat(corpus, task):
        get_raw_func = corpus_to_function.get(corpus)
        if get_raw_func is None:
            raise ValueError(f"Unknown corpus: {corpus}")

        train_text, train_label, test_text, _ = get_raw_func(task)
        train_texts_by_label = []
        unique_labels = set(train_label)
        for label in unique_labels:
            label_texts = [
                text for i, text in enumerate(train_text) if train_label[i] == label
            ]
            concat_text = "\n\n".join(label_texts)
            train_texts_by_label.append(concat_text)
        train_dist = [len(tokenize(t)) for t in train_texts_by_label]
        test_dist = [len(tokenize(t)) for t in test_text]
        print(
            f"{corpus} {task} training has # Avg. "
            f"{round(np.mean(train_dist))} (std. {round(np.std(train_dist))})."
        )
        print(
            f"{corpus} {task} testing has # Avg. "
            f"{round(np.mean(test_dist))} (std. {round(np.std(test_dist))})."
        )

    # RJ
    for task in ["control", "imitation", "obfuscation"]:
        _corpus_stat("rj", task)
    # EBG
    for task in ["no_protection", "imitation", "obfuscation"]:
        _corpus_stat("ebg", task)
    # LCMC v.1.1-Interview
    for task in ["no_protection"]:
        _corpus_stat("lcmc", task)


def load_rj(
    task: str, corpus_dir: str = "corpora/rj"
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Read in texts and labels from the Riddell-Juola (RJ) corpus. The RJ version contains
    a control group, and attacks of imitation and obfuscation.
    The control group will be used as if there is 'no protection' involved, simulating
    the cross-topic scenario of authorship attribution attacks.

    Args:
        task: One of ['control', 'imitation', 'obfuscation'], which corresponds to
            strategies 'no protection', '(manual) imitation',
            and '(manual) obfuscation', respectively.
        corpus_dir: Path to RJ corpus.

    Returns:
        Text/label of train/test sets.
    """
    if task in ["control", "imitation", "obfuscation"]:
        train_text, train_label, test_text, test_label = [], [], [], []
        authors = [
            f.name.split(".")[0]
            for f in os.scandir(os.path.join(corpus_dir, "attacks_" + task))
            if not f.name.startswith(".")
        ]
        # cfeec8 does not have training data
        for dir_ in [
            os.path.join(corpus_dir, author) for author in authors if author != "cfeec8"
        ]:
            for raw in os.scandir(dir_):
                train_text.append(open(raw.path, "r", encoding="utf8").read())
                train_label.append(raw.name.split("_")[0])
        # read in testing
        test_text, test_label = zip(
            *[
                (
                    open(
                        f.path,
                        "r",
                        encoding=chardet.detect(open(f.path, "rb").read())["encoding"],
                    ).read(),
                    f.name.split(".")[0],
                )
                for f in os.scandir(os.path.join(corpus_dir, "attacks_" + task))
                if ".txt" in f.name
            ]
        )
        return train_text, train_label, list(test_text), list(test_label)
    else:
        raise ValueError(f"Unknown task: {task}.")


def load_ebg(
    task: str, corpus_dir: str = "corpora/ebg"
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Read in texts and labels from the Extended Brennan-Greenstadt (EBG) corpus.
    Note, we conveniently use the first sample of each subject to obtain the no
    protection baseline; this differs from RJ as it has a real control group.
    The held-out samples will be used as if there is 'no protection' involved,
    simulating the topic-overlap scenario of authorship attribution attacks.

    Args:
        task: One of ['no_protection', 'imitation', 'obfuscation'], which corresponds
            strategies of no protection, 'imitation' (manual), and
            'obfuscation' (manual), respectively.
        corpus_dir: Path to the EBG corpus.

    Returns:
        Text/label of train/test sets.
    """
    if task not in ["no_protection", "imitation", "obfuscation"]:
        raise ValueError(f"Unknown task: {task}.")
    train_text, train_label, test_text, test_label = [], [], [], []

    for author in os.scandir(corpus_dir):
        if not author.name.startswith("."):
            if task == "no_protection":
                train_text_, test_text_ = [], []
                for f in os.scandir(author.path):
                    if re.match(r"[a-z]+_[0-9]{2}_*", f.name):
                        if f.name.endswith("_01_1.txt"):
                            test_text_.append(
                                open(
                                    f.path,
                                    "r",
                                    encoding=chardet.detect(open(f.path, "rb").read())[
                                        "encoding"
                                    ],
                                ).read()
                            )
                        else:
                            train_text_.append(
                                open(
                                    f.path,
                                    "r",
                                    encoding=chardet.detect(open(f.path, "rb").read())[
                                        "encoding"
                                    ],
                                ).read()
                            )
                train_text.extend(train_text_)
                test_text.extend(test_text_)
                train_label.extend([author.name] * len(train_text_))
                test_label.extend([author.name] * len(test_text_))
            else:
                train_text.extend(
                    [
                        open(
                            f.path,
                            "r",
                            encoding=chardet.detect(open(f.path, "rb").read())[
                                "encoding"
                            ],
                        ).read()
                        for f in os.scandir(author.path)
                        if re.match(r"[a-z]+_[0-9]{2}_*", f.name)
                    ]
                )
                train_label.extend(
                    [author.name]
                    * len(
                        [
                            f.name
                            for f in os.scandir(author.path)
                            if re.match(r"[a-z]+_[0-9]{2}_*", f.name)
                        ]
                    )
                )
            # read in testing
            if task == "imitation":
                test_text.extend(
                    [
                        open(
                            f.path,
                            "r",
                            encoding=chardet.detect(open(f.path, "rb").read())[
                                "encoding"
                            ],
                        ).read()
                        for f in os.scandir(author.path)
                        if re.match(r"[a-z]+_imitation_01.txt", f.name)
                    ]
                )
                test_label.append(author.name)
            if task == "obfuscation":
                test_text.extend(
                    [
                        open(
                            f.path,
                            "r",
                            encoding=chardet.detect(open(f.path, "rb").read())[
                                "encoding"
                            ],
                        ).read()
                        for f in os.scandir(author.path)
                        if re.match(r"[a-z]+_obfuscation.txt", f.name)
                    ]
                )
                test_label.append(author.name)
    return train_text, train_label, test_text, test_label


def load_lcmc(
    task: str = "no_protection", corpus_dir: str = "corpora/lcmc"
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Read in texts and labels from the Loyola Computer Mediated Communication (LCMC)
    corpus. The corpus contains various genres and topics, and is categorized into
    spoken and written modalities.

    We are only interested using written materials to fingerprint the interview sample.
    One of the interview samples will be chosen as if there is 'no protection' involved,
    simulating the cross-modal scenario of authorship attribution attacks.

    The subset created is named LCMC V.1.1-Interview, useful for cross-modal authorship
    attribution studies. It has an average training size of 8071 (std. deviation 1673)
    and an average testing size of 654 (std. deviation 228), which nicely mirrors the
    amounts of data used in RJ and EBG.

    The LCMC corpus follows specific file naming conventions for files of individual's
    text: email, essay, blog, interview, chat (individual contribution), discussion
    (individual contribution).
    The naming conventions are as follows: Sn or Snn X n Y n

    - one or 2 digit identifier for participant (with Subject renumbering)
    - genre identifier:
        E = Email
        S = Essay
        B = Blog
        C = Chat
        P = Phone Interview
        D = Discussion
    - For Phase 1 genres, order this genre encountered by the individual in phase 1
    (1, 2, or 3)
    - topic identifier:
        C = Catholic Church
        G = Gay Marriage
        I = War In Iraq
        M = Legalization of Marijuana
        P = Privacy Rights
        S = Sex Discrimination
    - order topic appeared in this genre for this individual

    Example: "S2P2S5" translates to: Second participant text; phone interview was second
    genre encountered (in Phase 1); sex discrimination was the fifth topic encountered
    in interviews.

    Args:
        task: Specifies the data retrieval task, fixed to 'cross_modal'.
        corpus_dir: Path to the LCMC corpus directory. Defaults to "corpora/lcmc".

    Returns:
        The text and labels of the training and testing sets.

    Notes:
        1. Two naming errors were corrected:
           - 'S21S2G4.txt' -> 'S21D2G4.txt'.
           - 'S1D113.txt' -> 'S1D1I3.txt'.
        2. The dataset is designed to be executed once. After the initial run with a
            seed of 42, a JSON file containing LCMC V1.1-Interview is generated in the
            `./corpora/lcmc/lcmc_v1.1_interview.json`
    """
    path_to_json = os.path.join(corpus_dir, f"lcmc_v1.1_interview.json")
    # if the subcorpus already exists
    if os.path.exists(path_to_json):
        lcmc_sub = json.load(open(path_to_json, "r"))
        print(f"Loading saved LCMC {task}.")

        return (
            lcmc_sub["train_text"],
            lcmc_sub["train_label"],
            lcmc_sub["test_text"],
            lcmc_sub["test_label"],
        )
    else:
        random.seed(42)
        data = []
        all_categories = [
            "Chat",
            "Discussion",
            "Interviews",  # speech transcribed
            "Blogs",
            "Emails",
            "Essays",
        ]  # written
        # read all samples
        for d in [
            os.path.join(d.path, "Correlated")
            for d in os.scandir(corpus_dir)
            if d.name in all_categories
        ]:
            for f in os.scandir(d):
                entry = {}
                file_name = f.name.split(".")[0]
                # regularize user numbering to 7
                file_name = (
                    file_name
                    if len(file_name) == 7
                    else file_name[:1] + "0" + file_name[1:]
                )
                entry["label"] = file_name[:3]
                entry["genre"] = file_name[3]
                entry["topic"] = file_name[5]
                entry["text"] = open(
                    f.path,
                    "r",
                    encoding=chardet.detect(open(f.path, "rb").read())["encoding"],
                ).read()
                data.append(entry)

        if task == "no_protection":  # -> LCMC-Spoken
            # using 3 written genres as training
            train_dicts = list(
                filter(lambda ent: ent["genre"] in ["E", "S", "B"], data)
            )
            # randomly choose 3 interview samples (concatenated) for each author as testing
            # that is, the topics are randomly chosen from 6 possible topics
            test_dicts = []
            for label in [
                "S{:02d}".format(num) for num in range(1, 22)
            ]:  # 21 candidates
                test_dicts.extend(
                    random.sample(
                        list(
                            filter(
                                lambda ent: (ent["label"] == label)
                                and (ent["genre"] == "P"),
                                data,
                            )
                        ),
                        k=1,
                    )
                )
        else:
            raise ValueError(f"Unknown task: {task}.")
        # unpack train and test sample dicts
        train_text = [d["text"] for d in train_dicts]
        train_label = [d["label"] for d in train_dicts]
        test_text, test_label = [], []
        for label in ["S{:02d}".format(num) for num in range(1, 22)]:
            test_label.append(label)
            test_text.append(
                "\n\n".join(
                    [
                        ent["text"]
                        for ent in list(
                            filter(lambda ent: ent["label"] in label, test_dicts)
                        )
                    ]
                )
            )
        # save it
        os.makedirs(corpus_dir, exist_ok=True)
        data_to_save = {
            "train_text": train_text,
            "train_label": train_label,
            "test_text": test_text,
            "test_label": test_label,
        }
        with open(path_to_json, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)
            print(f"LCMC V1.1-Interview is saved to {path_to_json}.")

    return train_text, train_label, test_text, test_label
