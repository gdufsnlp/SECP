# SECP
Code for Paper: [Paraphrase Makes Perfect: Leveraging Expression Paraphrase to Improve Implicit Sentiment Learning](https://aclanthology.org/2025.coling-main.245/)

## Overview
In this repository, we provide code for **S**entiment **E**xpression **C**onversion based **P**araphrase (SECP), which focuses on improving implicit sentiment learning in Aspect-based Sentiment Analysis.

![SECP_model](https://github.com/user-attachments/assets/f767a514-925f-4c69-991f-aacbc25bc347)

## Requirements
- cuda 11.4
- Python 3.10.12
  - PyTorch==2.0.1
  - Transformers==4.33.0
  - scikit-learn====1.3.0
  - openprompt==1.0.1
  - PyYAML==6.0.1
  - numpy==1.25.2
  - sentencepiece==0.1.96

## Datasets
To use the `implicit_sentiment` labeling provided by [SCAPT-ABSA](https://github.com/Tribleave/SCAPT-ABSA), we reformatted the data from [SemEval2014 Laptop/Restaurant]() following [ASGCN](https://github.com/GeneZC/ASGCN) and appended a label to each sample, which indicates whether it is an implicit sentiment expression ("Y" and "N" indicate the implicit and explicit sentiment expression respectively).
The data and paraphrased sentences for each datasets are provided in `data` and `data/paraphrased_data`.


## Usage
```bash
# Create a virtual environment and install necessary packages.
bash setup.sh
# Activate the virtual environment.
source .venv/bin/activate

# Train the models according to the configs (./config)
bash run.sh
```

Checkpoints are saved in `saved_checkpoints`.



## Citation
If you find this work useful, please cite the following:
```bibtex
@inproceedings{li-etal-2025-paraphrase,
    title = "Paraphrase Makes Perfect: Leveraging Expression Paraphrase to Improve Implicit Sentiment Learning",
    author = "Li, Xia  and
      Wang, Junlang  and
      Zheng, Yongqiang  and
      Chen, Yuan  and
      Zheng, Yangjia",
    editor = "Rambow, Owen  and
      Wanner, Leo  and
      Apidianaki, Marianna  and
      Al-Khalifa, Hend  and
      Eugenio, Barbara Di  and
      Schockaert, Steven",
    booktitle = "Proceedings of the 31st International Conference on Computational Linguistics",
    month = jan,
    year = "2025",
    address = "Abu Dhabi, UAE",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.coling-main.245/",
    pages = "3631--3647",
}
```
