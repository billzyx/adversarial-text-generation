import os
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from torcheval.metrics.functional import binary_auroc, binary_accuracy

import ppl_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cache_dir = None
base_model_dir_list = [
    './models',
]
for dir_path in base_model_dir_list:
    if os.path.isdir(dir_path):
        cache_dir = dir_path
        break


class PPLModel:
    def __init__(self, model_name, device=device, context_label=0):
        self.device = device
        self.context_label = context_label
        # print('Context label:', context_label)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, cache_dir=cache_dir, trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )

        model.eval()
        self.model = model
        if torch.cuda.device_count() > 1:
            self.device = 0

        self.user_prefix = ""
        self.user_assistant_infix = "\n"
        self.assistant_postfix = "\n"
        if self.tokenizer.chat_template:
            self.get_chat_prefix_postfix()

    def get_chat_prefix_postfix(self):
        chat_list = [{"role": "user", "content": "xxx"}, {"role": "assistant", "content": "yyy"}]
        text = self.tokenizer.apply_chat_template(chat_list, tokenize=False, add_generation_prompt=False)
        print(text)
        self.user_prefix = text.split('xxx')[0]
        self.user_assistant_infix = text.split('xxx')[1].split('yyy')[0]
        self.assistant_postfix = text.split('yyy')[1]

    def ppl_to_label(self, ppl, threshold, context_label=None):
        if context_label is None:
            context_label = self.context_label
        if context_label == 0:
            return 0 if ppl < threshold else 1
        elif context_label == 1:
            return 1 if ppl < threshold else 0
        else:
            raise (ValueError("Context label must be 0 or 1"))

    def cal_eer_threshold(self, label_list, predict_ppl_list, context_label=None):
        if context_label is None:
            context_label = self.context_label
        if context_label == 1:
            label_list = 1 - np.array(label_list)
        fpr, tpr, thresholds = metrics.roc_curve(label_list, predict_ppl_list)
        # Find the point where FPR and FNR are equal (EER point)
        fnr = 1 - tpr
        eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
        return eer_threshold

    def cal_ppl(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            loss = self.model(input_ids=inputs["input_ids"], labels=inputs["input_ids"]).loss
        ppl = torch.exp(loss).detach().cpu().numpy()
        ppl = float(ppl)
        return ppl

    def cal_ppl_token_level_list(self, text, context):
        token_list = self.tokenizer.tokenize(context)
        context_len = len(token_list)

        inputs = self.tokenizer(text + context, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            model_output = self.model(input_ids=input_ids)

        # Shift so that tokens < n predict n
        shift_logits = model_output.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).detach()

        # Cumulative sum and cumulative count for mean calculation
        cumulative_sum = torch.cumsum(loss, dim=0)
        cumulative_count = torch.arange(1, loss.size(0) + 1, dtype=torch.float, device=cumulative_sum.device)

        # Calculate cumulative mean
        cumulative_mean = cumulative_sum / cumulative_count
        ppl_list = torch.exp(cumulative_mean).detach().cpu().numpy().tolist()
        ppl_list = ppl_list[-context_len:]
        return ppl_list, token_list

    def cal_metrics(self, dataset, postfix_context, metric='auc'):
        hc_ppl_list = []
        ad_ppl_list = []
        for idx, data in enumerate(dataset):
            if data['label'] == self.context_label:
                hc_ppl_list.append(self.cal_ppl(data['text'] + postfix_context))
            else:
                ad_ppl_list.append(self.cal_ppl(data['text'] + postfix_context))
        label_list = np.concatenate([np.zeros(len(hc_ppl_list)), np.ones(len(ad_ppl_list))])
        predict_ppl_list = np.concatenate([hc_ppl_list, ad_ppl_list])

        metric_value = 0.
        metric_list = process_metrics(metric)

        all_metrics_dict = ppl_metrics.cal_all_metrics(label_list, predict_ppl_list)

        for metric_name in metric_list:
            metric_value += all_metrics_dict[metric_name]

        return metric_value

    def cal_all_metrics(self, dataset, postfix_context):
        hc_ppl_list = []
        ad_ppl_list = []
        for idx, data in enumerate(dataset):
            if data['label'] == self.context_label:
                hc_ppl_list.append(self.cal_ppl(data['text'] + postfix_context))
            else:
                ad_ppl_list.append(self.cal_ppl(data['text'] + postfix_context))
        label_list = np.concatenate([np.zeros(len(hc_ppl_list)), np.ones(len(ad_ppl_list))])
        predict_ppl_list = np.concatenate([hc_ppl_list, ad_ppl_list])
        return_dict = ppl_metrics.cal_all_metrics(label_list, predict_ppl_list)
        token_list = self.tokenizer.tokenize(postfix_context)
        context_len = len(token_list)
        return_dict['token_length'] = context_len
        return return_dict

    def regular_generate(self, dataset, postfix_instruction=None, token_length=100):
        output_list = []
        for data in dataset:
            text = data['text']
            if postfix_instruction is not None:
                text = text + postfix_instruction
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"]
            with torch.no_grad():
                model_output = self.model.generate(
                    input_ids=input_ids, do_sample=False, max_new_tokens=token_length,
                )
            output_text = self.tokenizer.decode(model_output[0][len(input_ids[0]):], skip_special_tokens=True)
            print(output_text)
            output_list.append(output_text)
        return output_list

    def generate(self, prompt, token_length=1000):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        with torch.no_grad():
            model_output = self.model.generate(
                input_ids=input_ids, do_sample=False, max_new_tokens=token_length,
            )
        output_text = self.tokenizer.decode(model_output[0][len(input_ids[0]):], skip_special_tokens=True)
        return output_text


    def lm_search(self, dataset, token_length=400, postfix_instruction=None, return_dict=False, stop_criterion='eos',
                  stop_by_score=False, stop_by_score_token_threshold=20, metric='auc', top_p=0.9,
                  regularization_label=0, truncate_max_sequence=True):
        print('lm search with top-p {}, regularization label {}, metric {}'.format(
            top_p, regularization_label, metric))
        stop_criterion = self.process_stop_criterion(stop_criterion)

        new_token_list = []
        max_score = self.cal_metrics(dataset, postfix_instruction, metric)
        score_list = [max_score]
        max_idx = -1
        for i in (pbar := tqdm(range(token_length))):
            next_tokens_seq_scores_list_hc = []
            next_tokens_seq_scores_list_ad = []
            next_tokens_regular_scores_list_hc = []
            next_tokens_regular_scores_list_ad = []

            for data in dataset:
                text = data['text']
                if postfix_instruction is not None:
                    text = text + postfix_instruction
                inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
                input_ids = inputs["input_ids"]
                if new_token_list:
                    content = self.tokenizer.decode(torch.cat(new_token_list, dim=-1)[0])
                    inputs = self.tokenizer(text + content, return_tensors="pt").to(self.device)
                    input_ids = inputs["input_ids"]
                with torch.no_grad():
                    model_output = self.model(input_ids=input_ids)

                next_tokens_seq_scores = self.cal_next_token_seq_score(model_output, input_ids)

                next_tokens_scores_regular = model_output.logits[:, -1, :].detach().clone()
                next_tokens_scores_regular = torch.softmax(next_tokens_scores_regular, dim=-1)

                if data['label'] == self.context_label:
                    next_tokens_seq_scores_list_hc.append(next_tokens_seq_scores)
                    next_tokens_regular_scores_list_hc.append(next_tokens_scores_regular)
                else:
                    next_tokens_seq_scores_list_ad.append(next_tokens_seq_scores)
                    next_tokens_regular_scores_list_ad.append(next_tokens_scores_regular)

            next_tokens_regular_scores_list = self.process_regularization_list(
                next_tokens_regular_scores_list_hc, next_tokens_regular_scores_list_ad, regularization_label
            )
            next_tokens_scores = cal_next_token_metrics(next_tokens_seq_scores_list_hc, next_tokens_seq_scores_list_ad,
                                                        next_tokens_regular_scores_list, top_p, metric)

            next_tokens = torch.argmax(next_tokens_scores, dim=-1)
            next_tokens_scores_max = float(torch.max(next_tokens_scores))
            new_token_list.append(next_tokens[:, None])
            if metric is not None:
                # text_output = self.tokenizer.batch_decode(torch.cat(new_token_list, dim=-1), skip_special_tokens=False)
                # print(text_output)
                score_list.append(next_tokens_scores_max)
                pbar.set_postfix_str('{}: {}'.format(metric, next_tokens_scores_max))
                if next_tokens_scores_max > max_score:
                    max_score = next_tokens_scores_max
                    max_idx = i
                elif stop_by_score:
                    if i - max_idx > stop_by_score_token_threshold:
                        break
            else:
                max_idx = i
            if stop_criterion is not None and i >= 1:
                if next_tokens[0] in stop_criterion:
                    print('Stop at token: {}, {}, at idx: {}'.format(
                        self.tokenizer.decode(next_tokens[0]), next_tokens[0], i
                    ))
                    new_token_list = new_token_list[:-1]
                    break

        print(max_idx, max_score)
        if truncate_max_sequence:
            if max_idx == -1:
                text_output = ""
            else:
                new_token_list = new_token_list[:max_idx + 1]
                text_output = self.tokenizer.batch_decode(torch.cat(new_token_list, dim=-1), skip_special_tokens=False)
                print(text_output)
                text_output = text_output[0]
        else:
            text_output = self.tokenizer.batch_decode(torch.cat(new_token_list, dim=-1), skip_special_tokens=False)
            print(text_output)
            text_output = text_output[0]

        if return_dict:
            return_dict = {
                'text': text_output,
                'postfix_context': postfix_instruction + text_output,
                'score': float(max_score),
                'score_list': score_list,
            }
            return_dict.update(self.cal_all_metrics(dataset, postfix_instruction + text_output))
            return return_dict

        return text_output

    def cal_next_token_seq_score(self, model_output, input_ids):
        # Shift so that tokens < n predict n
        shift_logits = model_output.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).detach()
        # Merge loss of existing token with new token
        next_tokens_scores = model_output.logits[:, -1, :].detach()
        next_tokens_scores = -torch.log(torch.softmax(next_tokens_scores, dim=-1))
        next_tokens_scores = torch.unsqueeze(next_tokens_scores, dim=-1)
        loss = loss.repeat(next_tokens_scores.size()[1], 1)
        loss = loss.unsqueeze(0)
        next_tokens_scores = torch.cat([next_tokens_scores, loss], dim=-1)
        next_tokens_scores = torch.mean(next_tokens_scores, dim=-1)
        next_tokens_scores = torch.exp(next_tokens_scores)
        return next_tokens_scores

    def process_stop_criterion(self, stop_criterion):
        if stop_criterion is not None:
            if isinstance(stop_criterion, str):
                stop_criterion = [stop_criterion]
            assert isinstance(stop_criterion, list)
            stop_criterion_list = []
            for single_stop_criterion in stop_criterion:
                if single_stop_criterion == 'eos':
                    stop_criterion_list.append(self.tokenizer.eos_token_id)
                    stop_criterion_list.extend(list(self.tokenizer.added_tokens_decoder.keys()))
                else:
                    stop_criterion_list.append(
                        self.tokenizer.encode(single_stop_criterion, add_special_tokens=False)[-1])
                    stop_criterion_list.append(self.tokenizer.convert_tokens_to_ids(single_stop_criterion))
            stop_criterion = stop_criterion_list
        return stop_criterion

    def process_regularization_list(self, next_tokens_regular_scores_list_label_0,
                                    next_tokens_regular_scores_list_label_1, regularization_label):
        assert regularization_label in [0, 1, 'all']
        if regularization_label == 0:
            return next_tokens_regular_scores_list_label_0
        elif regularization_label == 1:
            return next_tokens_regular_scores_list_label_1
        else:
            return next_tokens_regular_scores_list_label_0 + next_tokens_regular_scores_list_label_1

    def inference(self, text):
        completion = self.query_model([
            {"role": "user", "content": str(text)}
        ])
        return completion

    def inference_with_messages(self, messages):
        completion = self.query_model(messages)
        return completion

    def query_model(self, messages):
        input_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)
        gen_tokens = self.model.generate(
            input_ids,
            max_new_tokens=1000,
            do_sample=False,
            # temperature=0.7,
        )

        gen_text = self.tokenizer.decode(gen_tokens[0])
        # print(gen_text)

        gen_text_output_only = self.tokenizer.decode(gen_tokens[0][len(input_ids[0]):]).strip()
        # print(gen_text_output_only)
        messages.append({"role": "assistant", "content": gen_text_output_only})
        return messages


def calculate_eer_threshold_batch(predictions, labels):
    batch_size, num_samples = predictions.shape

    # Sort predictions and labels by prediction score within each batch
    sorted_indices = torch.argsort(predictions, dim=1, descending=True)
    sorted_predictions = torch.gather(predictions, 1, sorted_indices)
    sorted_labels = torch.gather(labels, 1, sorted_indices)

    # Calculate true positives and false positives
    cum_pos = torch.cumsum(sorted_labels, dim=1)
    cum_neg = torch.cumsum(1 - sorted_labels, dim=1)

    # Calculate TPR and FPR
    tpr = cum_pos / cum_pos[:, -1].unsqueeze(1)
    fpr = cum_neg / cum_neg[:, -1].unsqueeze(1)

    # Calculate FNR
    fnr = 1 - tpr

    # Find the EER threshold where FPR equals FNR
    eer_diff = torch.abs(fnr - fpr)
    eer_threshold_indices = torch.argmin(eer_diff, dim=1)
    eer_thresholds = sorted_predictions[torch.arange(batch_size), eer_threshold_indices]

    return eer_thresholds


def calculate_accuracy_with_eer_batch(predictions, labels):
    # Calculate the EER thresholds for each batch
    eer_thresholds = calculate_eer_threshold_batch(predictions, labels)
    # print(f"eer_thresholds: {eer_thresholds}")

    # Apply thresholds to predictions
    binary_predictions = (predictions >= eer_thresholds.unsqueeze(1)).float()

    # Calculate accuracy for each batch
    accuracy_list = batch_binary_accuracy(binary_predictions, labels)

    return accuracy_list


def batch_binary_accuracy(inputs, targets, threshold=0.5):
    """
    Compute binary accuracy for batches of predictions and targets.

    Parameters:
    inputs (Tensor) : Tensor of label predictions with shape of (batch_size, n_samples).
    targets (Tensor): Tensor of ground truth labels with shape of (batch_size, n_samples).
    threshold (float, default 0.5): Threshold for converting inputs into predicted labels for each sample.

    Returns:
    Tensor: Tensor containing binary accuracy for each batch.
    """
    # Apply threshold to inputs to get predicted labels
    preds = torch.where(inputs < threshold, 0, 1)

    # Compare predictions to targets and compute accuracy for each batch
    accuracies = (preds == targets).float().mean(dim=1)

    return accuracies


def process_metrics(metrics_str):
    metrics_list = metrics_str.split('+')
    available_metrics = ['acc', 'auc', 'ppl_score', 'ppl_diff']
    for metric in metrics_list:
        assert metric in available_metrics
    return metrics_list


def cal_next_token_metrics(next_tokens_seq_scores_list_label_0, next_tokens_scores_list_label_1,
                           next_tokens_scores_list_regular, top_p, metric='auc'):
    next_tokens_seq_scores_list_label_0 = torch.cat(next_tokens_seq_scores_list_label_0)
    next_tokens_scores_list_label_1 = torch.cat(next_tokens_scores_list_label_1)
    pred = torch.cat([next_tokens_seq_scores_list_label_0, next_tokens_scores_list_label_1], dim=0)
    pred = torch.transpose(pred, 1, 0)
    labels = torch.cat(
        [torch.zeros(len(next_tokens_seq_scores_list_label_0)), torch.ones(len(next_tokens_scores_list_label_1))]
    ).to(pred.device)
    labels = labels.repeat(pred.size()[0], 1)
    next_tokens_scores = torch.zeros(next_tokens_scores_list_label_1.size()[-1]).to(pred.device)
    metrics_list = process_metrics(metric)
    if 'acc' in metrics_list:
        next_tokens_scores += calculate_accuracy_with_eer_batch(pred, labels).clone()
    if 'auc' in metrics_list:
        next_tokens_scores += binary_auroc(pred, labels, num_tasks=len(labels))
    if 'ppl_score' in metrics_list:
        label_0_mean = torch.mean(next_tokens_seq_scores_list_label_0, dim=0)
        label_1_mean = torch.mean(next_tokens_scores_list_label_1, dim=0)
        std_0 = torch.std(next_tokens_seq_scores_list_label_0, unbiased=True, dim=0)
        std_1 = torch.std(next_tokens_scores_list_label_1, unbiased=True, dim=0)
        pooled_std = torch.sqrt(((std_0 ** 2 + std_1 ** 2) / 2))
        next_tokens_scores += ((- label_0_mean + label_1_mean) / pooled_std)
    if 'ppl_diff' in metrics_list:
        next_tokens_scores += (- torch.mean(next_tokens_seq_scores_list_label_0, dim=0) +
                               torch.mean(next_tokens_scores_list_label_1, dim=0))
    next_tokens_scores = next_tokens_scores.unsqueeze(0).clone()
    # top p
    next_tokens_scores_regular = torch.mean(torch.stack(next_tokens_scores_list_regular), dim=0)
    sorted_prob, sorted_indices = torch.sort(next_tokens_scores_regular, descending=True)
    cumulative_probs = torch.cumsum(sorted_prob, dim=-1)
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep at least one token
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    # Set removed token logits to a very large negative value (effectively zero probability)
    indices_to_remove = sorted_indices[sorted_indices_to_remove].unsqueeze(0)
    next_tokens_scores.scatter_(dim=-1, index=indices_to_remove, value=float('-inf'))
    return next_tokens_scores
