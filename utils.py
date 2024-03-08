"""
This module contains helper functions used to reproduce results in *Local Large language Models Are a Strong Defense
Against Authorship Attribution Attacks*.
"""

import os
import re
import csv
import json
import random
import chardet
import numpy as np
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from subprocess import check_output
from sklearn.svm import SVC
from typing import List, Tuple, NoReturn
from sklearn.pipeline import Pipeline
from transformers import T5TokenizerFast, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from writeprints_static import WriteprintsStatic
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import accuracy_score, roc_auc_score
from pytorch_lightning import seed_everything

SEED = 42

LLAMA_REWRITE_TASKS = ['llama_imitation_instruction_only',
                       'llama_imitation_500_tokens_example',
                       'llama_imitation_2000_tokens_example',
                       'llama_simplification_instruction_only',
                       'llama_simplification_voa_vocab',
                       'llama_obfuscation_instruction_only',
                       'llama_obfuscation_demographics',
                       'llama_obfuscation_persona']


def corpus_stat() -> None:
    """
    Compute and display statistics about the number of tokens in different corpora and tasks.
    The function uses the Moses tokenizer for tokenization and fetches data from specified
    corpus retrieval functions.

    Supported corpora include 'rj', 'ebg', and 'lcmc', and each has its own set of tasks.
    For each corpus and task, the function will output the average and standard deviation
    of the number of tokens in both training and testing data.

    The retrieval functions for the corpora should be pre-defined,
    and their format is exemplified by the `get_rj_raw` function.

    """
    from sacremoses import MosesTokenizer
    tokenize = lambda s: MosesTokenizer(lang="en").tokenize(s, escape=False)

    corpus_to_function = {'rj': get_rj_raw, 'ebg': get_ebg_raw,
                          # 'lcmc': get_lcmc_raw
                          }

    def _corpus_stat(corpus, task):
        get_raw_func = corpus_to_function.get(corpus)
        if get_raw_func is None:
            raise ValueError(f"Unknown corpus: {corpus}")

        train_text, train_label, test_text, _ = get_raw_func(task)
        train_texts_by_label = []
        unique_labels = set(train_label)
        for label in unique_labels:
            label_texts = [text for i, text in enumerate(train_text) if train_label[i] == label]
            concat_text = '\n\n'.join(label_texts)
            train_texts_by_label.append(concat_text)
        train_dist = [len(tokenize(t)) for t in train_texts_by_label]
        test_dist = [len(tokenize(t)) for t in test_text]
        print(f"{corpus} {task} training has # Avg. {round(np.mean(train_dist))} (std. {round(np.std(train_dist))}).")
        print(f"{corpus} {task} testing has # Avg. {round(np.mean(test_dist))} (std. {round(np.std(test_dist))}).")

    # RJ
    for task in ['control', 'imitation', 'obfuscation']:
        _corpus_stat('rj', task)
    # EBG
    for task in ['hold_one_out', 'imitation', 'obfuscation']:
        _corpus_stat('ebg', task)
    # for task in ['cross_modal', 'cross_genre', 'cross_topic']:  # LCMC-Spoken, LCMC-Email, and LCMC-Marihuana
    #     _corpus_stat('lcmc', task)


def get_rj_raw(task: str, corpus_dir: str = "corpora/rj") -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Read in texts and labels from the Riddell-Juola (RJ) corpus. The RJ version contains a control group, and attacks of
    imitation and obfuscation.

    Args:
        task: One of ['control', 'imitation', 'obfuscation'], which corresponds to strategies 'no protection',
    '(manual) imitation', and '(manual) obfuscation', respectively. corpus_dir: Path to RJ corpus.

    Returns:
        Text/label of train/test sets.
    """
    if task in ['control', 'imitation', 'obfuscation']:
        train_text, train_label, test_text, test_label = [], [], [], []
        authors = [f.name.split(".")[0]
                   for f in os.scandir(os.path.join(corpus_dir, "attacks_" + task)) if not f.name.startswith(".")]
        # cfeec8 does not have training data
        for dir_ in [os.path.join(corpus_dir, author) for author in authors if author != "cfeec8"]:
            for raw in os.scandir(dir_):
                train_text.append(open(raw.path, "r", encoding="utf8").read())
                train_label.append(raw.name.split("_")[0])
        # read in testing
        test_text, test_label = zip(
            *[(open(f.path, "r", encoding=chardet.detect(open(f.path, "rb").read())["encoding"], ).read(),
               f.name.split(".")[0],) for f in os.scandir(
                os.path.join(corpus_dir, "attacks_" + task)) if ".txt" in f.name])
        return train_text, train_label, list(test_text), list(test_label)
    else:
        raise ValueError(f'Unknown task: {task}.')


def get_ebg_raw(task: str, corpus_dir: str = "corpora/ebg") -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Read in texts and labels from the Extended Brennan-Greenstadt (EBG) corpus.

    Args:
        task: One of ['hold_one_out', 'imitation', 'obfuscation'], which corresponds strategies of no protection,
        'imitation' (manual), and 'obfuscation' (manual), respectively.
        Note, we conveniently use the first sample of each subject to obtain the no protection baseline; this differs
        from RJ as it has a real control group.
        corpus_dir: Path to the EBG corpus.

    Returns:
        Text/label of train/test sets.
    """
    if task not in ['hold_one_out', 'imitation', 'obfuscation']:
        raise ValueError(f'Unknown task: {task}.')
    train_text, train_label, test_text, test_label = [], [], [], []

    for author in os.scandir(corpus_dir):
        if not author.name.startswith("."):
            if task == 'hold_one_out':
                train_text_, test_text_ = [], []
                for f in os.scandir(author.path):
                    if re.match(r"[a-z]+_[0-9]{2}_*", f.name):
                        if f.name.endswith("_01_1.txt"):
                            test_text_.append(open(f.path, "r",
                                                   encoding=chardet.detect(open(f.path, "rb").read())[
                                                       "encoding"], ).read())
                        else:
                            train_text_.append(open(f.path, "r",
                                                    encoding=chardet.detect(open(f.path, "rb").read())[
                                                        "encoding"], ).read())
                train_text.extend(train_text_)
                test_text.extend(test_text_)
                train_label.extend([author.name] * len(train_text_))
                test_label.extend([author.name] * len(test_text_))
            else:
                train_text.extend([open(f.path, "r",
                                        encoding=chardet.detect(open(f.path, "rb").read())["encoding"], ).read()
                                   for f in os.scandir(author.path)
                                   if re.match(r"[a-z]+_[0-9]{2}_*", f.name)])
                train_label.extend([author.name] * len([f.name for f
                                                        in os.scandir(author.path)
                                                        if re.match(r"[a-z]+_[0-9]{2}_*", f.name)]))
            # read in testing
            if task == "imitation":
                test_text.extend(
                    [open(f.path, "r", encoding=chardet.detect(open(f.path, "rb").read())["encoding"]).read()
                     for f in os.scandir(author.path)
                     if re.match(r"[a-z]+_imitation_01.txt", f.name)])
                test_label.append(author.name)
            if task == "obfuscation":
                test_text.extend(
                    [open(f.path, "r", encoding=chardet.detect(open(f.path, "rb").read())["encoding"]).read()
                     for f in os.scandir(author.path) if re.match(r"[a-z]+_obfuscation.txt", f.name)])
                test_label.append(author.name)
    return train_text, train_label, test_text, test_label


def get_imdb62_raw(corpus_file: str = 'corpora/imdb62/imdb62.txt',
                   max_test_tokens: int = 1000) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Read the IMDB62 dataset from a specified file and splits the data into training and testing sets.
    A single test sample is extracted as if it's in a no protection scenario using sklearn's train_test_split.

    Parameters:
        corpus_file: The path to the file containing the IMDB62 dataset. Default is 'corpora/imdb62/imdb62.txt'.
        max_test_tokens: The maximum number of tokens for each test text. Default is 1000.

    Returns:
        Text/label of train/test sets.
    """

    with open(corpus_file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        corpus = [row for row in reader]
    # imdb62 has five fields: 'reviewId', 'userId', 'itemId', 'rating', 'title', and 'content'
    train_text, test_text, train_label, test_label = train_test_split(
        [r[-1] for r in corpus],  # we only use 'content'
        [r[1] for r in corpus],  # 'userId'
        test_size=62,  # Set the test_size to 62 to get one sample per label in the test set
        random_state=SEED,
        stratify=[r[1] for r in corpus]  # Stratify on the labels to ensure each label is represented
    )

    return train_text, train_label, [shorten_text(t, max_test_tokens) for t in test_text], test_label


#
# def get_lcmc_raw(task: str,
#                  hold_one_out: bool = False,
#                  corpus_dir: str = "corpora/lcmc",
#                  max_test_tokens: int = 1000) -> Tuple[List[str], List[str], List[str], List[str]]:
#     """
#     Read in texts and labels from the Loyola Computer Mediated Communication (LCMC) corpus. The corpus contains
#     various genres and topics, and is categorized into spoken and written modalities.
#
#     The LCMC corpus follows specific file naming conventions for files of individual's text: email, essay, blog,
#     interview, chat (individual contribution), discussion (individual contribution).
#     The naming conventions are as follows: Sn or Snn X n Y n
#
#     - one or 2 digit identifier for participant (with Subject renumbering)
#     - genre identifier:
#         E = Email
#         S = Essay
#         B = Blog
#         C = Chat
#         P = Phone Interview
#         D = Discussion
#     - For Phase 1 genres, order this genre encountered by the individual in phase 1 (1, 2, or 3)
#     - topic identifier:
#         C = Catholic Church
#         G = Gay Marriage
#         I = War In Iraq
#         M = Legalization of Marijuana
#         P = Privacy Rights
#         S = Sex Discrimination
#     - order topic appeared in this genre for this individual
#
#     Example: "S2P2S5" translates to: Second participant text; phone interview was second genre encountered (in Phase 1);
#     sex discrimination was the fifth topic encountered in interviews.
#
#     Args:
#         task: Specifies the data retrieval task. Valid options are 'cross_modal', 'cross_genre', or 'cross_topic'.
#         corpus_dir: Path to the LCMC corpus directory. Defaults to "corpora/lcmc".
#         hold_one_out: If return the first samples for no protection scenarios.
#         max_test_tokens: Limits the number of words in the `test_text` to the first approximately `max_test_tokens`
#         words, respecting sentence boundaries. Defaults to 1000.
#
#     Returns:
#         The text and labels of the training and testing sets.
#
#     Notes:
#         1. Two naming errors were corrected:
#            - 'S21S2G4.txt' -> 'S21D2G4.txt'.
#            - 'S1D113.txt' -> 'S1D1I3.txt'.
#         2. The function is designed to be executed once. After the initial run with a seed of 42, three subsets of the
#            corpus are created and saved as JSON files:
#            - LCMC-Spoken (cross_modal),
#            - LCMC-Email (cross_genre), and
#            - LCMC-Marihuana (cross_topic).
#         3. The tests samples are trunked to have ca. 1,000 tokens (defined by llama2 tokenizer) using shorten_text().
#     """
#     path_to_json = os.path.join(corpus_dir, f"lcmc_{task}.json")
#     # if a certain subcorpus already exists
#     if os.path.exists(path_to_json):
#         lcmc_sub = json.load(open(path_to_json, "r"))
#         print(f"Loading saved LCMC {task}.")
#         if not hold_one_out:
#             return lcmc_sub['train_text'], lcmc_sub['train_label'], lcmc_sub['test_text'], lcmc_sub['test_label']
#         else:
#             hold_one_out_train = defaultdict(list)
#             hold_one_out_test = defaultdict(list)
#             seen_labels = set()
#             for i, label in enumerate(lcmc_sub['train_label']):
#                 if label not in seen_labels:
#                     hold_one_out_test['test_text'].append(lcmc_sub['train_text'][i])
#                     hold_one_out_test['test_label'].append(label)
#                     seen_labels.add(label)
#                 else:
#                     hold_one_out_train['train_text'].append(lcmc_sub['train_text'][i])
#                     hold_one_out_train['train_label'].append(label)
#
#             return (hold_one_out_train['train_text'],
#                     hold_one_out_train['train_label'],
#                     [shorten_text(raw, max_test_tokens) for raw in hold_one_out_test['test_text']],
#                     hold_one_out_test['test_label'])
#     else:
#         random.seed(42)
#         data = []
#         all_categories = ['Chat', 'Discussion', 'Interviews',  # speech transcribed
#                           'Blogs', 'Emails', 'Essays']  # written
#         for d in [os.path.join(d.path, 'Correlated') for d in os.scandir(corpus_dir) if d.name in all_categories]:
#             for f in os.scandir(d):
#                 entry = {}
#                 file_name = f.name.split('.')[0]
#                 # regularize user numbering of length 7
#                 file_name = file_name if len(file_name) == 7 else file_name[:1] + '0' + file_name[1:]
#                 entry['label'] = file_name[:3]
#                 entry['genre'] = file_name[3]
#                 entry['topic'] = file_name[5]
#                 entry['text'] = open(f.path, "r", encoding=chardet.detect(open(f.path, "rb").read())["encoding"]).read()
#                 data.append(entry)
#
#         if task == 'cross_modal':  # -> LCMC-Spoken
#             # only using 3 written genres as training
#             train_dicts = list(filter(lambda ent: ent['genre'] in ['E', 'S', 'B'], data))
#             # randomly choose 6 samples from 3 spoken genres for each author
#             # concat 6 samples of the modality for each author in the very end
#             test_dicts = []
#             for label in ['S{:02d}'.format(num) for num in range(1, 22)]:
#                 test_dicts.extend(random.sample(list(filter(lambda ent: (ent['label'] == label)
#                                                                         and (ent['genre'] in ['C', 'P', 'D']), data)),
#                                                 6))
#         elif task == 'cross_genre':  # -> LCMC-Email
#             train_dicts = list(filter(lambda ent: ent['genre'] in ['S', 'B', 'C', 'P', 'D'], data))
#             test_dicts = list(filter(lambda ent: ent['genre'] in ['E'], data))
#         elif task == 'cross_topic':  # -> LCMC-Marijuana
#             train_dicts = list(filter(lambda ent: ent['topic'] in ['C', 'G', 'I', 'P', 'S'], data))
#             test_dicts = list(filter(lambda ent: ent['topic'] in ['M'], data))
#         else:
#             raise ValueError(f'Unknown task: {task}.')
#         # unpack train and test sample dicts
#         train_text = [d['text'] for d in train_dicts]
#         train_label = [d['label'] for d in train_dicts]
#         test_text, test_label = [], []
#         for label in ['S{:02d}'.format(num) for num in range(1, 22)]:
#             test_label.append(label)
#             test_text.append('\n\n'.join([ent['text'] for
#                                           ent in list(filter(lambda ent: ent['label'] in label, test_dicts))]))
#
#         test_text = [shorten_text(t, max_test_tokens) for t in test_text]
#
#         # save it
#         os.makedirs(corpus_dir, exist_ok=True)
#         data_to_save = {'train_text': train_text, 'train_label': train_label,
#                         'test_text': test_text, 'test_label': test_label}
#         with open(path_to_json, 'w', encoding='utf-8') as f:
#             json.dump(data_to_save, f, ensure_ascii=False, indent=4)
#             print(f"LCMC {task} is saved to {path_to_json}.")
#         return train_text, train_label, test_text, test_label

# todo: tokenizer decoding may introduce extra noise
def shorten_text(text: str, max_tokens: int) -> str:
    """
    Shortens the given text to a specified number of tokens,
    while ensuring that the text does not cut off in the middle of a sentence.

    Args:
        text: The text to be shortened.
        max_tokens: The maximum number of tokens for the shortened text. The returned sentence may be longer to respect
        sentence boundary.

    Returns:
        str: The shortened text.
    """
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
    tokens = tokenizer.tokenize(text)
    if len(tokens) <= max_tokens:
        return text  # return original text if it's shorter than max_tokens

    # initialize end_pos to max_tokens as a fallback
    end_pos = max_tokens
    # iterate through tokens after max_tokens to find the end of the sentence
    for i, token in enumerate(tokens[max_tokens - 1:], start=max_tokens - 1):
        # check for any of the sentence-ending characters
        if any(char in token for char in [".", "?", "!", "\n"]):
            end_pos = i
            break  # stop when the end of the sentence is found
    shortened_tokens = tokens[:end_pos + 1]
    shortened_text = tokenizer.convert_tokens_to_string(shortened_tokens)

    return shortened_text


# baselines
def backtranslation(raw: str) -> str:
    """
    Backtranslate text using translateLocally with the route from English to Garman then back to English.
    Note that `translateLocally` provides deterministic translation.

    We used 'translateLocally-Ubuntu-20.04.deb', downloaded from https://translatelocally.com/.
    Args:
        raw: The text to be back-translated.

    Returns:
        The back-translated text.
    """
    translator = "/usr/bin/translateLocally"  # path to local binary of translateLocally
    if not os.path.exists(translator):
        raise SystemExit(f"Path to translateLocally ({translator}) does not exist.")
    installed_models = {
        m.split(" ")[-1]
        for m in check_output([translator, "-l"], text=True).strip().split("\n")
    }
    for model in {"en-de-base", "de-en-base"} - installed_models:
        raise RuntimeError(
            f"The model {model} is not installed. Run '{translator} -d {model}'"
        )

    de_text = check_output([translator, "-m", "en-de-base"], input=raw, text=True, encoding='utf-8')
    en_text = check_output([translator, "-m", "de-en-base"], input=de_text, text=True, encoding='utf-8')

    return en_text.strip()


def style_transfer(raw: str,
                   bible_prefix: str = 'KJV: ',
                   device: str = 'cuda:0',
                   ckpt_path: str = './misc/bible_style_transfer_ckpt/epoch=0-step=305000-val_loss=1.028.ckpt') -> str:
    """
    Translate text into a specific style of the Bible. We choose the King Jame's Version (KJV) for its popularity and
    idiosyncratic style.

    Args:
        raw: The text to be translated.
        bible_prefix: Prefix to be added to the text for style transfer.
        device: Device to be used for computation.
        ckpt_path: Path to the checkpoint file.

    Returns:
        The text translated into Bible style.
    """
    # Load the model and tokenizer
    seed_everything(SEED)
    from bible_style_transfer import BibleStyleTransferModel
    model = BibleStyleTransferModel.load_from_checkpoint(ckpt_path).to(device)
    model.freeze()
    tokenizer = T5TokenizerFast.from_pretrained('google/t5-efficient-base-nl24')

    bible_style_list = []
    for sent in sent_tokenize(raw):
        bible_text = bible_prefix + sent
        encoding = tokenizer(
            bible_text,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt').to(device)

        decoded_ids = model.model.generate(
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask'],
            max_new_tokens=256,
            top_p=.9,
            temperature=1.0,
            do_sample=True)
        bible_style_list.append(tokenizer.decode(decoded_ids[0], skip_special_tokens=True).strip())
    bible_style_text = ' '.join(bible_style_list)

    return bible_style_text


def obfuscate_style(corpus: str, task: str, save_to_dir: str = 'baselines') -> NoReturn:
    """
    Generate and save adversarial samples given a corpus and a task (which corresponds to an approach camouflaging one's
    style). This is the heavy-lifting function of the experiments.

    Args:
        corpus: The name of the corpus, one of 'rj' or 'ebg'.
        task: The task to perform.
              For all corpora, task is one of
                    - automatic baselines: 'backtranslation' and 'style_transfer'
                    - llama rewrite:
                        - obfuscation: 'llama_obfuscation_instruction_only', 'llama_obfuscation_demographics', and
                            'llama_obfuscation_persona'
                        - imitation: 'llama_imitation_instruction_only', 'llama_imitation_500_tokens_example', and
                            'llama_imitation_2000_tokens_example'
                        - simplification: 'llama_simplification_instruction_only' and 'llama_simplification_voa_vocab'
              For some manual baselines: TODO
        save_to_dir: The directory to save the obfuscated samples.

    Returns:
        None; save a json file containing three lists: the obfuscated text, their original version, and the test labels.
    """
    random.seed(SEED)
    # result container
    output_text_label = {}
    prompts = []
    if corpus == 'rj':
        _, _, test_text, test_label = get_rj_raw("control")
    elif corpus == 'ebg':
        _, _, test_text, test_label = get_ebg_raw("obfuscation")  # for ebg, task doesn't matter here
    # elif corpus == 'lcmc_spoken':
    #     _, _, test_text, test_label = get_lcmc_raw("cross_modal")
    # elif corpus == 'lcmc_email':
    #     _, _, test_text, test_label = get_lcmc_raw("cross_genre")
    # elif corpus == 'lcmc_marijuana':
    #     _, _, test_text, test_label = get_lcmc_raw("cross_topic")
    # todo: more corpora
    else:
        raise ValueError(f'Unknown corpus: {corpus}.')

    # two strong baselines: backtranslation and bible style transfer
    if task == 'backtranslation':
        obfuscated_text = [backtranslation(raw)
                           for raw in tqdm(test_text,
                                           desc=f"Backtranslate {corpus} sentences with translateLocally (EN-DE-EN)")]
    elif task == 'style_transfer':
        obfuscated_text = [style_transfer(raw) for raw in tqdm(test_text,
                                                               desc=f"Translate {corpus} sentence into KJV style")]
    # methods using llm rewrites
    # prompts for simplification and imitation are fixed
    # prompts for two obfuscation tasks involve sampling from the BIG5 corpus
    elif task in LLAMA_REWRITE_TASKS:
        from llama_rewrite import llama_rewrite, llama_prompt_safety_check, build_prompt
        # make sure sys/instruction relevant tokens not present in the text to-be-obfuscated
        for raw in test_text:
            llama_prompt_safety_check(raw)
        prompts = [build_prompt(task, raw) for raw in test_text]
        obfuscated_text = [llama_rewrite(prompt)
                           for prompt in tqdm(prompts,
                                              desc=f"Llama rewrites {corpus} sentences using {task} strategy")]
    else:
        raise ValueError(f'Unknown task: {task}.')

    # if save_to_dir:
    output_text_label.update({"obfuscated_text": obfuscated_text,
                              "test_text": test_text,
                              "test_label": test_label,
                              "prompts": prompts})
    with open(os.path.join(f'{save_to_dir}', f'{corpus}_{task}.json'), 'w', encoding='utf-8') as f:
        json.dump(output_text_label, f, ensure_ascii=False, indent=4)


# threat models related
def vectorize_writeprints_static(raws: List[str]) -> np.ndarray:
    """
    Extract `writeprints-static` features from a list of texts. Done by library `writeprints-static` v0.0.2.
    Args:
        raws: a list of str.
    Returns:
         writeprints-static vector.
    """
    vec = WriteprintsStatic()
    features = vec.transform(raws)

    return features.toarray()


def get_llama_rewrites(corpus: str,
                       task: str,
                       return_original: bool = False,
                       output_dir: str = 'outputs/llama_rewrites'):
    json_file = os.path.join(output_dir, f"{corpus}_{task}.json")
    output = json.load(open(json_file, 'r'))
    test_text = output['obfuscated_text']
    test_label = output['test_label']
    # if corpus == 'rj':
    #     train_text, train_label, _, _ = get_rj_raw('control')
    # elif corpus == 'ebg':
    #     train_text, train_label, _, _ = get_ebg_raw('hold_one_out')  # for ebg, task doesn't matter here
    # elif corpus == 'lcmc_spoken':
    #     train_text, train_label, _, _ = get_lcmc_raw('cross_modal')
    # elif corpus == 'lcmc_email':
    #     train_text, train_label, _, _ = get_lcmc_raw('cross_genre')
    # elif corpus == 'lcmc_marijuana':
    #     train_text, train_label, _, _ = get_lcmc_raw('cross_topic')
    # else:
    #     raise ValueError(f'Unknown corpus {corpus}.')

    if not return_original:
        return test_text, test_label
    else:
        original_text = output['test_text']
        return test_text, test_label, original_text


def calculate_writeprints_performance(corpus: str, task: str):
    if corpus == 'rj':
        if task in ['control', 'obfuscation', 'imitation']:
            train_text, train_label, test_text, test_label = get_rj_raw(task)
        elif task in LLAMA_REWRITE_TASKS:
            train_text, train_label, _, _ = get_rj_raw('control')
            test_text, test_label = get_llama_rewrites(corpus, task)
        else:
            raise ValueError(f'Unknown {task} for corpus {corpus}.')
    elif corpus == 'ebg':
        if task in ['hold_one_out', 'obfuscation', 'imitation']:
            train_text, train_label, test_text, test_label = get_ebg_raw(task)
        elif task in LLAMA_REWRITE_TASKS:
            train_text, train_label, _, _ = get_ebg_raw('obfuscation')  # task does not matter for training
            test_text, test_label = get_llama_rewrites(corpus, task)
        else:
            raise ValueError(f'Unknown {task} for corpus {corpus}.')
    # todo: lcmc relevant tasks
    elif corpus == 'imdb62':
        ...
    else:
        raise ValueError(f'Unknown {task} for corpus {corpus}.')
    print(
        f'Writeprints-static (SVC) on {corpus} {task} has an accuracy of {ml_pipeline(train_text, train_label, test_text, test_label):.3f}.')
    print(f'The task has {len(set(train_label))} candidates (random guessing {1 / len(set(train_label)):.3f}).')


def ml_pipeline(train_text, train_label, test_text, test_label):
    pipeline = Pipeline(
        [("normalizer", Normalizer(norm="l1")),
         ("scaler", StandardScaler()),
         ("svm", SVC(kernel='linear', tol=1e-6))])

    x_train = vectorize_writeprints_static(train_text)
    x_test = vectorize_writeprints_static(test_text)
    pipeline.fit(x_train, train_label)

    return accuracy_score(y_true=test_label, y_pred=pipeline.predict(x_test))
