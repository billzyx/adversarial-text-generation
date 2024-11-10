# Adversarial Text Generation using Large Language Models for Dementia Detection (EMNLP 24)

[Paper Link](https://aclanthology.org/2024.emnlp-main.1222/)

## Usage Guide

1. Download data from [talk bank](https://dementia.talkbank.org/ADReSS-2020/)

2. Install environment

    ```
    conda env create -f environment.yml
    ```

3. Run scripts

    ```
    python3 run_ppl.py
    ```


Cite:

```
@inproceedings{zhu-etal-2024-adversarial,
    title = "Adversarial Text Generation using Large Language Models for Dementia Detection",
    author = "Zhu, Youxiang  and
      Lin, Nana  and
      Balivada, Kiran  and
      Haehn, Daniel  and
      Liang, Xiaohui",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.1222",
    pages = "21918--21933",
    abstract = "Although large language models (LLMs) excel in various text classification tasks, regular prompting strategies (e.g., few-shot prompting) do not work well with dementia detection via picture description. The challenge lies in the language marks for dementia are unclear, and LLM may struggle with relating its internal knowledge to dementia detection. In this paper, we present an accurate and interpretable classification approach by Adversarial Text Generation (ATG), a novel decoding strategy that could relate dementia detection with other tasks. We further develop a comprehensive set of instructions corresponding to various tasks and use them to guide ATG, achieving the best accuracy of 85{\%}, {\textgreater}10{\%} improvement compared to the regular prompting strategies. In addition, we introduce feature context, a human-understandable text that reveals the underlying features of LLM used for classifying dementia. From feature contexts, we found that dementia detection can be related to tasks such as assessing attention to detail, language, and clarity with specific features of the environment, character, and other picture content or language-related features. Future work includes incorporating multi-modal LLMs to interpret speech and picture information.",
}
```