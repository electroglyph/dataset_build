import argparse
import os
import sys
import glob
from transformers import AutoTokenizer, AutoConfig
import humanize
import json


def get_text(folder: str) -> list[str]:
    """Return list of strings, one string for each txt file in folder

    Args:
        folder (str): folder with txt files

    Returns:
        list[str]: list of the txt files
    """
    retval = []
    files = glob.glob(folder + "/*.txt")
    for f in files:
        with open(f, "r") as of:
            retval.append(of.read())
    return retval


def get_langs() -> dict:
    """Get dict of languages and their folders

    Returns:
        dict: {lang_name: directory}
    """
    langs = {}
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    stuff = os.listdir(here)
    coding = False
    for s in stuff:
        if s == "coding":
            coding = True
            continue
        p = os.path.join(here, s)
        if os.path.isdir(p):
            langs[s] = p
    if coding:
        coding = os.listdir(os.path.join(here, "coding"))
        for c in coding:
            p = os.path.join(here, "coding", c)
            if os.path.isdir(p):
                langs[c] = p
    return langs


def parse_list(input: str) -> list:
    if not input:
        raise ValueError("Empty argument")
    if not "," in input:
        return [input.strip()]
    else:
        return [i.strip() for i in input.split(",")]


def main(
    include: list,
    exclude: list,
    model: str,
    chat: bool,
    autosplit: bool,
    tokenize: bool,
):
    lang_dirs = get_langs()
    if include:
        include_dirs = {}
        for i in include:
            d = lang_dirs.get(i)
            if d:
                include_dirs[i] = d
            else:
                print(f"Error: invalid language included: {i}, check --list")
                sys.exit(1)
        lang_dirs = include_dirs
    if exclude:
        for e in exclude:
            d = lang_dirs.pop(e, None)
            if not d:
                print(
                    f"Error: invalid language excluded: {e}, check --list (or included languages)"
                )
                sys.exit(1)
    tokenizer = None
    count = 0
    if model:
        tokenizer = AutoTokenizer.from_pretrained(model)
        config = AutoConfig.from_pretrained(model)
        tf = config.get_text_config()
        max_len = 0
        if tf:
            tf = tf.to_dict()
            max_len = tf.get("max_position_embeddings")
        if tokenizer.model_max_length and max_len:
            max_len = min(tokenizer.model_max_length, max_len)
        else:
            max_len = max_len or tokenizer.model_max_length
        unk = tokenizer.unk_token_id
        to_exclude = []
        for lang, dir in lang_dirs.items():
            # print(f"Checking if model can tokenize language: {lang} ...")
            texts = get_text(dir)
            for t in texts:
                tokens = tokenizer(t)
                count += len(tokens["input_ids"])
                if unk in tokens["input_ids"]:
                    to_exclude.append(lang)
                    print(
                        f"Unknown token ({tokenizer.unk_token}) found for lang: {lang}, skipping this language."
                    )
                    break
        if not to_exclude:
            print("Model supports all languages.")
        else:
            print(
                f"Total of {len(to_exclude)} {"language" if len(to_exclude) == 1 else "languages"} removed."
            )
            for te in to_exclude:
                del lang_dirs[te]
        output = ""
        for k, v in lang_dirs.items():
            print(f"Grabbing txt files from: {v}")
            texts = get_text(v)
            for t in texts:
                if chat:
                    convo = [
                        {
                            "role": "user",
                            "content": f"Write a bunch of stuff in this language, which is either an ISO 639-3 language code or a programming language: {k}",
                        },
                        {
                            "role": "assistant",
                            "content": t,
                        },
                    ]
                    convo = tokenizer.apply_chat_template(
                        convo, tokenize=False, add_generation_prompt=False
                    )
                    output = f"{output}{convo}"
                else:
                    output = f"{output}{t}"
        if autosplit:
            split_output = []
            tokenized_output = tokenizer(
                output,
                max_length=max_len,
                truncation=True,
                return_overflowing_tokens=True,
                stride=0,
            )
            if tokenize:
                split_output = [
                    {"input_ids": ids, "attention_mask": mask}
                    for ids, mask in zip(
                        tokenized_output["input_ids"],
                        tokenized_output["attention_mask"],
                    )
                ]
                with open("output.json", "w") as f:
                    f.write(json.dumps(split_output))
                    print("Finished, list of dicts written to output.json")
            else:
                for to in tokenized_output["input_ids"]:
                    split_output.append(tokenizer.decode(to))
                with open("output.json", "w") as f:
                    f.write(json.dumps(split_output))
                    print("Finished, output written to output.json")
        else:
            if tokenize:
                tokenized = tokenizer(output)
                with open("output.json", "w") as f:
                    f.write(json.dumps(dict(tokenized)))
                    print("Finished, dict written to output.json")
            else:
                with open("output.txt", "w") as f:
                    f.write(output)
                    print("Finished, output written to output.txt")
    else:
        output = ""
        for _, v in lang_dirs.items():
            print(f"Grabbing txt files from: {v}")
            texts = get_text(v)
            for t in texts:
                output = f"{output}{t}"
        with open("output.txt", "w") as f:
            f.write(output)
            print("Finished, output written to output.txt")
    if model:
        print(f"Maximum sequence length for model: {humanize.intcomma(max_len)}")
        print(f"Tokens processed: {humanize.intcomma(count)} (approximately)")


def run():
    parser = argparse.ArgumentParser(
        prog="dataset_build",
        description="Build a multilingual dataset for imatrix or quantization calibration of LLMs or embedding models",
    )
    parser.register("type", "string_list", parse_list)
    parser.add_argument(
        "-i",
        "--include",
        type="string_list",
        default=None,
        help="Comma separated list of languages to include, all languages are included by default.",
    )
    parser.add_argument(
        "-e",
        "--exclude",
        type="string_list",
        default=None,
        help="Comma separated list of languages to exclude, no languages are excluded by default.",
    )
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="List available languages and exit.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Path or name of HF model to use to check for unknown tokens.",
    )
    parser.add_argument(
        "-c",
        "--chat",
        action="store_true",
        help="Apply chat template to dataset, disabled by default. Requires model argument.",
    )
    parser.add_argument(
        "-a",
        "--autosplit",
        action="store_true",
        help="Output json file of array of strings, disabled by default. Each array will be less than or equal to maximum model sequence length. Requires model argument.",
    )
    parser.add_argument(
        "-t",
        "--tokenize",
        action="store_true",
        help="Output token ids instead of text, disabled by default. Requires model argument.",
    )
    args = parser.parse_args()
    if args.list:
        lang_dirs = get_langs()
        print(f"Languages: {",".join(lang_dirs.keys())}")
        sys.exit(0)
    if args.chat and not args.model:
        parser.error("--chat requires --model argument")
        sys.exit(1)
    if args.autosplit and not args.model:
        parser.error("--autosplit requires --model argument")
        sys.exit(1)
    if args.tokenize and not args.model:
        parser.error("--tokenize requires --model argument")
        sys.exit(1)
    main(
        args.include, args.exclude, args.model, args.chat, args.autosplit, args.tokenize
    )
