## dataset_build

This is a simple little tool to build a multilingual dataset for creating a llama.cpp imatrix, or for LLM/embedding model analysis pre-quantization.

## How it works

In the data folder there are a bunch of folders. Most of them match ISO 639-3 language codes. Any .txt files within will be added to the dataset. There is another subfolder in there named coding. It has subfolders named after programming languages. Any .txt files within them will be added to the dataset.

If you specify a model, languages will be passed thru the model's tokenizer to check for unknown tokens. If any are found, that language will be rejected.

## Included languages (so far)

Human languages (about 30,000 words each): Arabic, Bengali, Bulgarian, Czech, Danish, German, Greek, Estonian, Finnish, French, Hebrew, Hindi, Croatian, Hungarian, Indonesian, Italian, Japanese, Korean, Latvian, Lithuanian, Dutch, Norwegian, Polish, Portuguese, Romanian, Russian, Slovakian, Slovenian, Spanish, Serbian, Swahili, Swedish, Thai, Turkish, Ukrainian, Vietnamese, and Chinese

Programming languages (about 1,000 lines each): Bash, C#, C++, Dart, Elixir, Go, Haskell, HTML/CSS/Javascript, Java, Kotlin, Lua, Objective-C, Perl, PHP, Python, R, Ruby, Rust, SQL, Swift, and Typescript

Total tokens (so far): about 1.5 million

## What's the source of the data?

It's purely synthetic, sourced from Gemini 2.5 Pro. To generate data for the human languages I used this prompt: 

Write 30,000 words of varied text in the language I specify. Make sure it's 30,000 words. I'm creating a dataset which will be used to create an importance matrix of an LLM. I need the text to touch on a wide variety of topics. Write 3000 words each about these ten subjects: Scientific and Technical Disciplines; Medical and Life Sciences; Mathematics and Logic; Arts and Humanities; Fantasy, Mythology, and Folklore; Niche Hobbies and Obscure Knowledge; Jargon and Specialized Professional Language; Abstract and Conceptual Topics; Creative and Imaginative Writing Prompts; and Emerging and Interdisciplinary Fields. Try to allocate an approximately equal number of words to each different subject. The language I want you to write in is: {language}

To generate data for the programming languages I used this prompt:

Write 1000 lines of code in the programming language I specify. I'm creating a dataset to help me create an importance matrix for an LLM. Make the code varied and try to touch on everything possible in the specified language. This may or may not include unsafe code, metaprogamming, templates, structs, classes, etc. All the code needs to be in the same source file. The language I need code for is: {language}

I also added an additional English text file with this prompt:

Pick 1000 obscure english words and use each in a short sentence. output one sentence per line, no formatting

---

If you think I could've done a better job, please open a PR. If you want to add a language, please open a PR. If you want to improve on anything...don't be shy, open a PR.

## Installation

`pip install dataset_build`

## Usage

`dataset_build -h` :
```
usage: dataset_build [-h] [-i INCLUDE] [-e EXCLUDE] [-l] [-m MODEL] [-c] [-a] [-t] [--trust]

Build a multilingual dataset for imatrix or quantization calibration of LLMs or embedding models

options:
  -h, --help            show this help message and exit
  -i, --include INCLUDE
                        Comma separated list of languages to include, all languages are included by default.
  -e, --exclude EXCLUDE
                        Comma separated list of languages to exclude, no languages are excluded by default.
  -l, --list            List available languages and exit. If model is specified, count the tokens for each language as well.
  -m, --model MODEL     Path or name of HF model to use to check for unknown tokens.
  -c, --chat            Apply chat template to dataset, disabled by default. Requires model argument.
  -a, --autosplit       Output json file of list of strings, disabled by default. Each list will be less than or equal to maximum model sequence length. Requires model argument.
  -t, --tokenize        Output tokenizer output instead of text, disabled by default. Requires model argument.
  --trust               Set trust_remote_code=True on tokenizer and config. Requires model argument.
```

Unless you use --autosplit or --tokenize (or --list) a file called output.txt will be generated which is all the included languages stuffed into one text file.

If you use autosplit, the max sequence length for the model will be grabbed from either the model config or the tokenizer, and then the output will be a JSON file (named output.json) of a list of dicts. The dicts will contain whatever the tokenizer output is.

If you tokenize the output but don't autosplit, the output will be a JSON file (named output.json) of one dict with the tokenizer output.

If you use --model argument with --list, the tokenizer will be used to list the token counts for each language and the total tokens overall (for that tokenizer).

If you apply the chat template, each text file is stuffed into a conversation like this:

```python
convo = [
    {
        "role": "user",
        "content": f"Write a bunch of stuff in this language, which is either an ISO 639-3 language code or a programming language: {lang}",
    },
    {
        "role": "assistant",
        "content": text_file,
    },
]
```