import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from openprompt.prompts import ManualTemplate, SoftVerbalizer
from openprompt.plms import load_plm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from utils import DEVICE, PTDataset, train_collate_fn, test_collate_fn, \
    get_llm_generated_text, read_yaml, set_seed, construct_soft_label_values


def extract_at_mask(outputs: torch.Tensor, loss_ids: torch.Tensor):
    outputs = outputs[torch.where(loss_ids > 0)]
    outputs = outputs.view(loss_ids.shape[0], -1, outputs.shape[1])
    if outputs.shape[1] == 1:
        outputs = outputs.view(outputs.shape[0], outputs.shape[2])

    return outputs


def sentiment_consistent_contrastive_learning(
        original_mask_logits: torch.Tensor,
        additional_mask_logits: torch.Tensor,
        tau: float,
        labels: torch.Tensor,
        implicit_labels,
        soft_label_values
):

    # expanded_logits_matrix: [2 * batch_size, vocab_size]
    expanded_logits_matrix = torch.cat((original_mask_logits, additional_mask_logits), dim=0)

    # set the homologous_ids: [0, 1, ..., 2 * batch_size]
    # homologous_ids are the symbols for marking the homologous samples
    homologous_ids = torch.arange(0, expanded_logits_matrix.shape[0])
    for item in homologous_ids:
        if item < int(expanded_logits_matrix.shape[0] / 2):
            item += int(expanded_logits_matrix.shape[0] / 2)
        else:
            item -= int(expanded_logits_matrix.shape[0] / 2)

    # construct the positives and negatives according to labels and implicit_labels
    # store the 2 * batch value
    cl_labels_dimension = expanded_logits_matrix.shape[0]
    cl_labels = torch.zeros((cl_labels_dimension, cl_labels_dimension)).to(DEVICE)

    temp_labels = torch.cat((labels, labels), dim=0)
    reversed_implicit_labels = torch.ones(implicit_labels.shape[0]) - implicit_labels
    temp_implicit_labels = torch.cat((implicit_labels, reversed_implicit_labels), dim=0)

    for i in range(cl_labels_dimension):
        for j in range(cl_labels_dimension):
            # ignore current sample and set the homologous id
            if j == i:
                continue
            elif j == homologous_ids[i]:
                cl_labels[i][j] = 1.0

            else:
                first_id = int(np.abs((temp_labels[j] - temp_labels[i]).cpu().detach().numpy()))
                second_id = int(np.abs((temp_implicit_labels[j] - temp_implicit_labels[i]).cpu().detach().numpy()))
                cl_labels[i][j] = soft_label_values[first_id][second_id]

    # regularize the cl_labels to the probabilities of classes
    cl_labels = cl_labels.softmax(dim=-1)

    # similarities: [2 * batch_size, 2 * batch_size]
    similarities = F.cosine_similarity(
        x1=expanded_logits_matrix.unsqueeze(1),
        x2=expanded_logits_matrix.unsqueeze(0),
        dim=2
    )

    similarities = similarities - (torch.eye(expanded_logits_matrix.shape[0])).to(DEVICE)
    similarities = similarities / tau
    loss = F.cross_entropy(similarities, cl_labels)

    return loss


def evaluate(model, validate_dataloader: DataLoader, verbalizer):

    model.eval()

    # for measuring the metrics
    n_correct, n_total = 0, 0
    n_implicit_correct, n_implicit_total = 0, 0
    targets_all, outputs_all = None, None

    with torch.no_grad():
        for batch in tqdm(validate_dataloader):

            # get the input of model
            input_ids, attention_mask, loss_ids, labels, implicit_labels = batch.values()
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)

            mlm_model_output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            # current_batch_logits: [batch_size, max_len, vocab_size]
            current_batch_logits = mlm_model_output['logits']

            outputs = verbalizer.gather_outputs(mlm_model_output)

            # capture the hidden states of the mask tokens
            # outputs_at_mask: [batch_size, vocab_size]
            outputs_at_mask = extract_at_mask(outputs=outputs, loss_ids=loss_ids).to(DEVICE)
            mapped_logits = verbalizer.process_outputs(outputs_at_mask, batch=batch).to(DEVICE)

            # calculate the metrics of the validation model
            n_correct += (torch.argmax(mapped_logits, -1) == labels).sum().item()
            n_total += current_batch_logits.shape[0]

            # calculate the MF1 metric
            # store the true labels and prediction labels
            if targets_all is None:
                targets_all = labels
                outputs_all = mapped_logits
            else:
                targets_all = torch.cat((targets_all, labels), dim=0)
                outputs_all = torch.cat((outputs_all, mapped_logits), dim=0)

            # calculate the metrics of implicit examples
            implicit_examples_labels = []
            implicit_logits = []
            for i in range(len(implicit_labels)):

                # ignore the explicit examples
                if implicit_labels[i] == 0:
                    continue

                else:
                    # calculate the accuracy (the calculation of MF1 is at the following step)
                    implicit_examples_labels.append(labels[i])
                    implicit_logits.append(mapped_logits[i])

                    current_correct_label = int(labels[i])
                    n_implicit_total += 1

                    # the prediction of current example
                    current_prediction = int(torch.argmax(mapped_logits[i]))
                    if current_prediction == current_correct_label:
                        n_implicit_correct += 1

        # the normal metrics
        acc = n_correct / n_total
        f1 = f1_score(
            y_true=targets_all.cpu(),
            y_pred=torch.argmax(outputs_all, -1).cpu(),
            labels=[0, 1, 2],
            average='macro'
        )

        # the metrics for implicit examples
        implicit_acc = n_implicit_correct / n_implicit_total

        return acc, f1, implicit_acc


def train(
        model,
        train_dataloader: DataLoader,
        validate_dataloader: DataLoader,
        optimizer,
        schedular,
        verbalizer,
        loss_fct,
        config,
        loss_weights,
        optimizer_for_verbalizer=None,
):
    # init the variable "global_steps" which is constructed by "epoch_num * batch_num"
    global_steps = 0

    # init the max accuracy, f1-score for outputting the best result and save the optimal continuous prompt tokens
    max_val_acc, max_val_f1, max_val_epoch = 0, 0, 0

    for i_epoch in range(config['epoch_num']):
        print("-" * 100)
        print("Epoch: {}".format(i_epoch))
        n_correct, n_total, loss_total = 0, 0, 0

        model.train()

        # get the weights of corresponding losses
        csa_weight = loss_weights['csa_weight']
        cl_weight = loss_weights['cl_loss_weight']

        for batch in tqdm(train_dataloader):

            global_steps += 1

            # clear the gradient accumulators
            optimizer.zero_grad()
            if optimizer_for_verbalizer is not None:
                optimizer_for_verbalizer.zero_grad()

            # get the input of model
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            additional_input_ids = batch["additional_input_ids"].to(DEVICE)
            additional_attention_mask = batch["additional_attention_mask"].to(DEVICE)
            implicit_labels = batch["implicit_labels"]

            loss_ids = batch["loss_ids"]
            additional_loss_ids = batch["additional_loss_ids"]

            # get the batch size from the shape of input data
            batch_size = labels.shape[0]

            mlm_model_output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

            additional_mlm_model_output = model(
                input_ids=additional_input_ids,
                attention_mask=additional_attention_mask,
                output_hidden_states=True
            )

            # outputs, additional_outputs: [batch_size, seq_len, hidden_size]
            outputs = verbalizer.gather_outputs(mlm_model_output)
            additional_outputs = verbalizer.gather_outputs(additional_mlm_model_output)

            # capture the indices of each [MASK] token
            outputs_at_mask = extract_at_mask(outputs=outputs, loss_ids=loss_ids).to(DEVICE)
            additional_outputs_at_mask = extract_at_mask(outputs=additional_outputs, loss_ids=additional_loss_ids).to(
                DEVICE)

            # cls_from_outputs: [batch_size, hidden_size]
            cls_from_outputs = outputs[:, 0, :]

            # contrastive
            soft_label_values = construct_soft_label_values(config)
            cl_loss = sentiment_consistent_contrastive_learning(
                original_mask_logits=outputs_at_mask,
                additional_mask_logits=additional_outputs_at_mask,
                tau=config['cl_tau'],
                labels=labels,
                implicit_labels=implicit_labels,
                soft_label_values=soft_label_values
            )

            # label words mapping for the original outputs
            mapped_logits = verbalizer.process_outputs(outputs_at_mask, batch=batch).to(DEVICE)

            kl_loss = F.kl_div(
                input=F.log_softmax(cls_from_outputs / config['kd_tau'], dim=-1),
                target=F.softmax(outputs_at_mask / config['kd_tau'], dim=-1),
                reduction='batchmean'
            )
            csa_loss = kl_loss * config['kd_tau'] ** 2

            # get the losses of MLM task
            current_batch_loss = loss_fct(mapped_logits, labels)
            loss = current_batch_loss + (csa_weight * csa_loss) + (cl_weight * cl_loss)
            loss.backward()

            optimizer.step()
            if optimizer_for_verbalizer is not None:
                optimizer_for_verbalizer.step()
            if schedular is not None:
                schedular.step()

            n_total += batch_size
            loss_total += loss.item() * batch_size
            n_correct += (torch.argmax(mapped_logits, -1) == labels).sum().item()

            if global_steps % 100 == 0:
                train_acc = n_correct / n_total
                train_loss = loss_total / n_total
                print('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))

                # show the composition of "losses"
                print('original CE loss: {0}'.format(current_batch_loss))
                print('CL loss: {0}, weight: {1}'.format(cl_loss, cl_weight))
                print('CSA loss: {0}, weight: {1}'.format(csa_loss, csa_weight))

        val_acc, val_f1, implicit_acc = evaluate(
            model=model,
            validate_dataloader=validate_dataloader,
            verbalizer=verbalizer
        )
        current_epoch_loss = loss_total / n_total
        print('> the training loss of this epoch is : {:.4f}'.format(current_epoch_loss))
        print('> val_acc: {:.4f}, val_f1: {:.4f}, val_ISE: {:.4f}'.format(val_acc, val_f1, implicit_acc))

        if val_acc > max_val_acc and val_f1 > max_val_f1:
            max_val_acc = val_acc
            max_val_f1 = val_f1
            max_val_epoch = i_epoch

        # saving the fine-tuned model which perform better
        if val_acc >= max_val_acc and val_f1 >= max_val_f1:
            print("Saving fine-tuned model....")
            saved_model_path = os.path.join(config['output_dir'],
                                            'saved_epoch_{0}_acc_{1}_f1_{2}_{3}_model.bin'.format(i_epoch, val_acc,
                                                                                                  val_f1,
                                                                                                  config['dataset']))
            torch.save(model.state_dict(), saved_model_path)

            print("Saving trained soft_verbalizer...")
            saved_verbalizer_path = os.path.join(
                config['output_dir'],
                'saved_epoch_{0}_acc_{1}_f1_{2}_{3}_verbalizer.bin'.format(i_epoch, val_acc, val_f1,
                                                                           config['dataset'])
            )
            torch.save(verbalizer.state_dict(), saved_verbalizer_path)

        if i_epoch - max_val_epoch >= config['patience']:
            print(">> early stop.")
            break


def main(args, config):

    # get the path of dataset
    train_path, test_path = config['train_data_path'], config['test_data_path']

    # init the dataset
    train_dataset = PTDataset(path=train_path, implicit_symbol=True)
    test_dataset = PTDataset(path=test_path, implicit_symbol=True)

    # get the paraphrased sentences
    llm_generated_texts = get_llm_generated_text(path=config['llm_generated_path'])

    # load the PLM and its config
    bert_mlm, tokenizer, model_config, WrapClass = load_plm(
        model_name='bert',
        model_path=config['plm_dir']
    )
    bert_mlm = bert_mlm.to(DEVICE)

    # get the wrapped_tokenizer
    wrapped_tokenizer = WrapClass(
        tokenizer=tokenizer,
        max_seq_length=config['max_len']
    )

    # init the prompt templates for training
    manual_template = ManualTemplate(
        tokenizer=tokenizer,
        text=config['input_template_text']
    )

    additional_manual_template = ManualTemplate(
        tokenizer=tokenizer,
        text=config['additional_template_text']
    )

    verbalizer = SoftVerbalizer(
        tokenizer=tokenizer,
        model=bert_mlm,
        classes=['negative', 'neutral', 'positive'],
    ).to(DEVICE)

    # insert the paraphrased sentences for current item
    for i in range(len(train_dataset)):
        train_dataset.pt_datasets[i]['input_example'].meta = {"llm_generated_text": llm_generated_texts[i]}

    # init the collect function
    train_collect_fc = lambda samples: train_collate_fn(
        samples=samples,
        wrapped_tokenizer=wrapped_tokenizer,
        template=manual_template,
        template_for_llm_generated_text=additional_manual_template,
    )

    test_collect_fc = lambda samples: test_collate_fn(
        samples=samples,
        wrapped_tokenizer=wrapped_tokenizer,
        template=manual_template,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=train_collect_fc
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=test_collect_fc
    )

    # show the performance before training
    if args.do_eval_first:
        zero_shot_acc, zero_shot_f1, zero_shot_implicit_acc = evaluate(
            model=bert_mlm,
            validate_dataloader=test_dataloader,
            verbalizer=verbalizer
        )
        print("The performance before training is: ")
        print("ACC: ", round(zero_shot_acc, 5), " MF1: ", round(zero_shot_f1, 5), " ISE: ", round(zero_shot_implicit_acc, 5))

    # set the training config
    loss_fct = nn.CrossEntropyLoss(label_smoothing=config['ce_label_smoothing'])

    loss_weights = {
        'csa_weight': torch.tensor(config['csa_weight']),
        'cl_loss_weight': torch.tensor(config['cl_loss_weight'])
    }

    # set the optimizer and the trainable params
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_mlm.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': config['weight_decay']},
        {'params': [p for n, p in bert_mlm.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
    ]
    optimizer = AdamW(params=optimizer_grouped_parameters, lr=config['lr'])

    # set the optimizer for soft verbalizer
    optimizer_grouped_parameters_2 = [
        {'params': verbalizer.group_parameters_1, 'lr': config['soft_verbalizer_lr_1']},
        {'params': verbalizer.group_parameters_2, 'lr': config['soft_verbalizer_lr_2']}
    ]
    optimizer_for_verbalizer = AdamW(optimizer_grouped_parameters_2)

    # optional: set the warmup
    if args.do_warmup:
        training_total_steps = config['epoch_num'] * len(train_dataloader)
        schedular = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config['warmup_ratio'] * training_total_steps,
            num_training_steps=training_total_steps
        )
    else:
        schedular = None

    # begin to train
    train(
        model=bert_mlm,
        train_dataloader=train_dataloader,
        validate_dataloader=test_dataloader,
        optimizer=optimizer,
        schedular=schedular,
        verbalizer=verbalizer,
        loss_fct=loss_fct,
        config=config,
        loss_weights=loss_weights,
        optimizer_for_verbalizer=optimizer_for_verbalizer,
    )


if __name__ == '__main__':

    # setting arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path')
    parser.add_argument('--do_eval_first', action='store_true')
    parser.add_argument('--do_warmup', action='store_true')

    args = parser.parse_args()
    config = read_yaml(path=args.config_path)
    set_seed(config['seed'])

    # print the config dictionary
    print("The hyper-parameters are set as follow: ")
    print(config)

    main(
        args=args,
        config=config
    )
