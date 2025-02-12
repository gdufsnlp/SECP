import os
import random
from typing import Tuple, List, Dict

import torch
import numpy as np
import yaml

from torch.utils.data import Dataset
from openprompt.data_utils import InputExample

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed_value: int):

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed_value)


class PTDataset(Dataset):

    def __init__(self, path: str, implicit_symbol: bool):
        super(PTDataset, self).__init__()
        self.pt_datasets, self.polarities = self.read_corpus(
            path=path,
            implicit_symbol=implicit_symbol
        )

        # load all aspects in the dataset
        aspects = []
        for input_example_dict in self.pt_datasets:
            aspects.append(input_example_dict['input_example'].text_b)

        self.aspect_list = aspects
        self.aspect_set = list(set(aspects))

        # load the implicit labels (if given the "implicit symbol")
        if implicit_symbol:
            implicit_labels = []

            for current_input_example in self.pt_datasets:
                implicit_labels.append(current_input_example['implicit_label'])

            self.implicit_labels = implicit_labels

    def __getitem__(self, index):
        return self.pt_datasets[index], self.polarities[index]

    def __len__(self):
        return len(self.polarities)

    def read_corpus(self, path, implicit_symbol) -> Tuple[List[Dict], List[int]]:
        """
        Read the data of ABSA dataset and capture the List of InputExample of OpenPrompt package
        :param path: the path of dataset
        :param implicit_symbol: the symbol which denotes whether loading the dataset with implicit labels
        :return:
        """

        datasets = []
        polarities = []

        with open(path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        # handle the situation that loading the dataset with implicit labels
        if implicit_symbol:
            iteration = range(0, len(lines), 4)
            guid_divisor = 4
        else:
            iteration = range(0, len(lines), 3)
            guid_divisor = 3

        for i in iteration:
            # init a dictionary to store current data sample
            current_data = {}

            # get the context, which consists of two parts: the left context and right context of the aspect term
            text_left, _, text_right = [s.strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].strip()
            polarity = lines[i + 2].strip()

            # revert the whole sentence by text_left, text_right and aspect term
            full_context = text_left + ' ' + aspect + ' ' + text_right
            full_context = full_context.strip()

            if implicit_symbol:
                # store the current implicit label if passing the "implicit_symbol" arg
                implicit_label = lines[i + 3].strip()
                # the implicit label will be only 'Y' or 'N', so it's necessary to convert them to the int 1 or 0
                if implicit_label == 'Y':
                    current_data['implicit_label'] = 1
                elif implicit_label == 'N':
                    current_data['implicit_label'] = 0

            # change the type of polarity to integer
            polarity = int(polarity) + 1

            # construct the current input_example and store to the dict
            if i <= 0:
                current_input_example = InputExample(
                    guid=i,
                    text_a=full_context,
                    text_b=aspect,
                    label=polarity
                )
            else:
                current_input_example = InputExample(
                    guid=int(i / guid_divisor),
                    text_a=full_context,
                    text_b=aspect,
                    label=polarity
                )
            current_data['input_example'] = current_input_example

            # store to the corresponding list
            datasets.append(current_data)
            polarities.append(polarity)

        return datasets, polarities


def read_yaml(path):
    with open(path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file.read())


def test_collate_fn(
        samples: Tuple[Dict, int],
        wrapped_tokenizer,
        template,
) -> Dict:
    # get the input_example and polarities
    input_example_dicts, polarities = zip(*samples)

    # each item is a dictionary; Additionally, init a list for storing implicit labels
    model_inputs, implicit_labels = [], []

    # wrap and tokenize
    for item in input_example_dicts:

        implicit_labels.append(item['implicit_label'])

        tokenized_example = wrapped_tokenizer.tokenize_one_example(
            template.wrap_one_example(item['input_example']), teacher_forcing=False
        )

        # append the input_ids, attention_mask, loss_ids of the original input example
        model_inputs.append(tokenized_example)

    input_ids = []
    attention_mask = []
    loss_ids = []

    for current_dict in model_inputs:
        input_ids.append(current_dict['input_ids'])
        attention_mask.append(current_dict['attention_mask'])
        loss_ids.append(current_dict['loss_ids'])

    # covert them into tensor
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    loss_ids = torch.tensor(loss_ids)
    labels = torch.tensor(list(polarities))

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_ids": loss_ids,
        "labels": labels,
        "implicit_labels": implicit_labels
    }


def train_collate_fn(
        samples: Tuple[Dict, int],
        wrapped_tokenizer,
        template,
        template_for_llm_generated_text,
) -> Dict:
    # get the input_example and polarities
    input_example_dicts, polarities = zip(*samples)

    # each item is a dictionary
    model_original_inputs = []
    model_augmented_inputs = []

    # init a list for storing implicit labels
    implicit_labels = []

    # wrap and tokenize
    for item in input_example_dicts:

        # first, get the implicit labels from current sample
        implicit_labels.append(item['implicit_label'])

        tokenized_example = wrapped_tokenizer.tokenize_one_example(
            template.wrap_one_example(item['input_example']), teacher_forcing=False
        )

        tokenized_llm_generated_text = wrapped_tokenizer.tokenize_one_example(
            template_for_llm_generated_text.wrap_one_example(item['input_example']), teacher_forcing=False
        )

        model_original_inputs.append(tokenized_example)
        model_augmented_inputs.append(tokenized_llm_generated_text)

    # construct the input_example_tensor
    input_ids, attention_mask, loss_ids = [], [], []

    for current_dict in model_original_inputs:
        input_ids.append(current_dict['input_ids'])
        attention_mask.append(current_dict['attention_mask'])
        loss_ids.append(current_dict['loss_ids'])

    # construct the additional_input_tensor
    additional_input_ids, additional_attention_mask, additional_loss_ids = [], [], []

    for current_dict in model_augmented_inputs:
        additional_input_ids.append(current_dict['input_ids'])
        additional_attention_mask.append(current_dict['attention_mask'])
        additional_loss_ids.append(current_dict['loss_ids'])

    # covert them into tensor
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    loss_ids = torch.tensor(loss_ids)
    additional_input_ids = torch.tensor(additional_input_ids)
    additional_attention_mask = torch.tensor(additional_attention_mask)
    additional_loss_ids = torch.tensor(additional_loss_ids)
    labels = torch.tensor(list(polarities))
    implicit_labels = torch.tensor(implicit_labels)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_ids": loss_ids,
        "additional_input_ids": additional_input_ids,
        "additional_attention_mask": additional_attention_mask,
        "additional_loss_ids": additional_loss_ids,
        "labels": labels,
        "implicit_labels": implicit_labels
    }


def get_llm_generated_text(path) -> List[str]:
    llm_generated_texts = []
    with open(path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

        for line in lines:
            llm_generated_texts.append(line.strip())

    return llm_generated_texts


def construct_soft_label_values(config: Dict):
    soft_label_values = [[0, 0] for i in range(3)]
    soft_label_values[0][0] = config['alpha_1_beta_1']
    soft_label_values[0][1] = config['alpha_1_beta_n_1']
    soft_label_values[1][0] = config['alpha_n_1_beta_1']
    soft_label_values[1][1] = config['alpha_n_1_beta_n_1']
    soft_label_values[2][0] = config['alpha_n_1_beta_1']
    soft_label_values[2][1] = config['alpha_n_1_beta_n_1']

    return soft_label_values
