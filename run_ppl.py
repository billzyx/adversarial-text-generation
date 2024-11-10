import os
import torch
from scipy.stats import ttest_ind
import numpy as np
from sklearn.metrics import classification_report
import json
import pandas as pd
import re
import random

import dataloaders
import models
import ppl_metrics

random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data_id = 'ADReSS20-train-transcript'
test_data_id = 'ADReSS20-test-transcript'
print(train_data_id, test_data_id)

model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
print(model_name)

model = models.PPLModel(model_name=model_name)
result_dir = 'result_{}'.format(train_data_id)

result_instruction_dir = os.path.join(result_dir, 'result_instruction')
question_dir = os.path.join(result_dir, "questions")


def get_context_ppl_dataset(context_text, dataset):
    for idx, data in enumerate(dataset):
        text = data['text']
        ppl_list, token_list = model.cal_ppl_token_level_list(text, context_text)
        dataset[idx]['ppl'] = ppl_list[-1]
        dataset[idx]['ppl_list'] = ppl_list
        dataset[idx]['token_list'] = token_list
    return dataset


def dataset_to_ppl_list(dataset):
    hc_ppl_list = []
    ad_ppl_list = []

    for idx, data in enumerate(dataset):
        if data['label'] == 0:
            hc_ppl_list.append(dataset[idx]['ppl'])
        else:
            ad_ppl_list.append(dataset[idx]['ppl'])
    return hc_ppl_list, ad_ppl_list


def add_prefix_context_dataset(dataset, prefix_instruction=''):
    for idx, data in enumerate(dataset):
        dataset[idx]['text'] = prefix_instruction + data['text']
    return dataset


def show_single(postfix_context, prefix_context, save_dir=None, save_tag='default', score_list=None):
    if save_dir:
        save_dir = os.path.join(save_dir, model_name, save_tag)
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'prefix_context.txt'), 'w', encoding='utf-8') as f:
            f.write(prefix_context)
        with open(os.path.join(save_dir, 'postfix_context.txt'), 'w', encoding='utf-8') as f:
            f.write(postfix_context)
        if score_list:
            with open(os.path.join(save_dir, 'score_list.txt'), 'w', encoding='utf-8') as f:
                f.write(str(score_list))

    train_dataset = [data for data in dataloaders.load_dataset_easy(train_data_id)]
    test_dataset = [data for data in dataloaders.load_dataset_easy(test_data_id)]

    train_dataset = add_prefix_context_dataset(train_dataset, prefix_context)
    test_dataset = add_prefix_context_dataset(test_dataset, prefix_context)

    example_text = test_dataset[0]['text'] + postfix_context
    if save_dir:
        with open(os.path.join(save_dir, 'example.txt'), 'w', encoding='utf-8') as f:
            f.write(example_text)

    report_dict_train, report_dict_test = show_single_with_dataset(postfix_context, train_dataset, test_dataset,
                                                                   save_dir)

    return report_dict_train, report_dict_test


def show_single_with_dataset(postfix_context, train_dataset, test_dataset, save_dir=None):
    train_dataset = get_context_ppl_dataset(postfix_context, train_dataset)
    hc_ppl_list, ad_ppl_list = dataset_to_ppl_list(train_dataset)
    t_statistic, p_value = t_test(ad_ppl_list, hc_ppl_list)
    if save_dir:
        save_ppl(hc_ppl_list, ad_ppl_list, t_statistic, p_value, save_path=os.path.join(save_dir, 'train_ppl.json'))

    test_dataset = get_context_ppl_dataset(postfix_context, test_dataset)
    hc_ppl_list, ad_ppl_list = dataset_to_ppl_list(test_dataset)
    t_statistic, p_value = t_test(ad_ppl_list, hc_ppl_list)
    if save_dir:
        save_ppl(hc_ppl_list, ad_ppl_list, t_statistic, p_value, save_path=os.path.join(save_dir, 'test_ppl.json'))

    if save_dir:
        performance_dict_list = []
        for i in range(len(train_dataset[0]['ppl_list'])):
            report_dict_train, report_dict_test, wrong_data_list = cal_performance(
                train_dataset, test_dataset, ppl_list_index=i
            )
            performance_dict_list.append({
                'train_acc': report_dict_train['accuracy'],
                'train_auc': report_dict_train['auc'],
                'train_ppl_score': report_dict_train['ppl_score'],
                'train_ppl_diff': report_dict_train['ppl_diff'],
                'test_acc': report_dict_test['accuracy'],
                'test_auc': report_dict_test['auc'],
                'test_ppl_score': report_dict_test['ppl_score'],
                'test_ppl_diff': report_dict_test['ppl_diff'],
                'token': train_dataset[0]['token_list'][i],
            })
        pd.DataFrame(performance_dict_list).to_csv(os.path.join(save_dir, 'performance_dict.csv'))

    report_dict_train, report_dict_test, wrong_data_list = cal_performance(train_dataset, test_dataset)

    print(report_dict_train)
    print(report_dict_test)
    if save_dir:
        with open(os.path.join(save_dir, 'wrong.json'), 'w', encoding='utf-8') as fp:
            json.dump(wrong_data_list, fp)
        with open(os.path.join(save_dir, 'train_report.json'), 'w', encoding='utf-8') as fp:
            json.dump(report_dict_train, fp)
        with open(os.path.join(save_dir, 'test_report.json'), 'w', encoding='utf-8') as fp:
            json.dump(report_dict_test, fp)
    return report_dict_train, report_dict_test


def cal_performance(train_dataset, test_dataset, ppl_list_index=None):
    train_label = []
    train_ppl = []
    for data in train_dataset:
        train_label.append(data['label'])
        train_ppl.append(data['ppl'] if ppl_list_index is None else data['ppl_list'][ppl_list_index])
    eer_threshold = model.cal_eer_threshold(train_label, train_ppl)
    if ppl_list_index is None:
        print(eer_threshold)
    train_label_predict = []
    for ppl in train_ppl:
        train_label_predict.append(model.ppl_to_label(ppl, eer_threshold))
    report_dict_train = classification_report(train_label, train_label_predict, digits=4, output_dict=True)
    report_dict_train.update(ppl_metrics.cal_all_metrics(train_label, train_ppl))
    test_label = []
    test_ppl = []
    test_label_predict = []
    wrong_data_list = []
    for data in test_dataset:
        test_label.append(data['label'])
        ppl = data['ppl'] if ppl_list_index is None else data['ppl_list'][ppl_list_index]
        test_ppl.append(ppl)
        predict_label = model.ppl_to_label(ppl, eer_threshold)
        test_label_predict.append(predict_label)
        if predict_label != data['label']:
            wrong_data_list.append(data)
    report_dict_test = classification_report(test_label, test_label_predict, digits=4, output_dict=True)
    report_dict_test.update(ppl_metrics.cal_all_metrics(test_label, test_ppl))
    return report_dict_train, report_dict_test, wrong_data_list


def save_ppl(hc_ppl_list, ad_ppl_list, t_statistic, p_value, save_path):
    ppl_dict = dict()
    ppl_dict['hc_ppl_list'] = list(hc_ppl_list)
    ppl_dict['ad_ppl_list'] = list(ad_ppl_list)
    ppl_dict['hc_mean'] = float(np.mean(hc_ppl_list))
    ppl_dict['ad_mean'] = float(np.mean(ad_ppl_list))
    ppl_dict['t_statistic'] = t_statistic
    ppl_dict['p_value'] = p_value
    with open(save_path, 'w', encoding='utf-8') as fp:
        json.dump(ppl_dict, fp)


def t_test(ad_ppl_list, hc_ppl_list):
    print(np.mean(hc_ppl_list), np.mean(ad_ppl_list))
    t_statistic, p_value = ttest_ind(hc_ppl_list, ad_ppl_list)
    print("t-statistic: ", t_statistic)
    print("p-value: ", p_value)
    return t_statistic, p_value


def load_human_defined_contexts_dict(context_type):
    contexts_dir = "contexts"
    contexts_json_path = os.path.join(contexts_dir, context_type + '.json')
    with open(contexts_json_path, 'r', encoding='utf-8') as f:
        contexts_dict = json.load(f)
    # contexts_dict = {k: v + '\n' for k, v in contexts_dict.items()}
    return contexts_dict


def get_question_key(question_str):
    question_str = question_str.lower()
    question_str = re.sub(r'[^a-zA-Z]', '_', question_str)
    return question_str[:200]


def load_question_dict(question_generation_key, model_name_for_question=model_name):
    question_json_path = os.path.join(question_dir, model_name_for_question, question_generation_key,
                                      'aspect_instruction.json')
    if not os.path.isfile(question_json_path):
        return {}
    with open(question_json_path, 'r', encoding='utf-8') as f:
        question_list = json.load(f)
    question_list = sorted(list(set(question_list)))
    question_dict = {get_question_key(q): '{}'.format(q) for q in question_list}
    return question_dict


def extract_questions(text):
    text = text.replace('*', '')
    question_lines = re.findall(r'^\d+\..*', text, re.MULTILINE)

    processed_lines = []
    for line in question_lines:
        match = re.search(r'[a-zA-Z]', line)
        if match:
            start_index = match.start()
            processed_line = line[start_index:] if start_index > 0 else line
            processed_line = processed_line.replace('*', '')

            # if ":" in processed_line:
            #     processed_line = processed_line.split(":", maxsplit=1)[1]

            processed_line = processed_line.strip()
            processed_lines.append(processed_line)

    return list(set(processed_lines))


def standard_train(prefix_context='', postfix_instruction='', save_tag='default'):
    prefix_context = model.user_prefix + prefix_context
    print(prefix_context)
    train_dataset = [data for data in dataloaders.load_dataset_easy(train_data_id)]
    train_dataset = add_prefix_context_dataset(train_dataset, prefix_context)

    postfix_instruction = '\n' + postfix_instruction + model.user_assistant_infix
    print(postfix_instruction)
    postfix_content = model.lm_search(train_dataset, token_length=1000, postfix_instruction=postfix_instruction,
                                      stop_criterion='eos', stop_by_score=False,
                                      metric='ppl_score+acc', regularization_label=0, top_p=0.9)

    postfix_context = postfix_instruction + postfix_content
    print(postfix_context)
    show_single(postfix_context, prefix_context, save_dir=result_instruction_dir, save_tag=save_tag)


def generate_diff(question_generation_key):
    train_dataset = [data for data in dataloaders.load_dataset_easy(train_data_id)]

    train_dataset_hc = [data for data in train_dataset if data['label'] == 0]
    train_dataset_ad = [data for data in train_dataset if data['label'] == 1]

    train_dataset_ad_curr = train_dataset_ad
    train_dataset_hc_curr = train_dataset_hc

    diff_dataset = []
    for data_hc in train_dataset_hc_curr:
        for data_ad in train_dataset_ad_curr:
            text = "{}Text 1: {}\nText 2: {}\n".format(model.user_prefix, data_hc['text'], data_ad['text'])
            diff_dataset.append({'text': text})
    print(len(diff_dataset))

    diff_prompt = "Find out the difference between text 1 and text 2. Discuss the differences in a list of aspects."
    postfix_instruction = '\n' + diff_prompt + model.user_assistant_infix
    print(postfix_instruction)

    postfix_content_list = model.regular_generate(diff_dataset, token_length=1000,
                                                  postfix_instruction=postfix_instruction, )

    diff_dict = {
        'diff_list': postfix_content_list,
    }
    save_path = os.path.join(
        question_dir, model_name, question_generation_key, 'diff.json'
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding='utf-8') as json_file:
        json.dump(diff_dict, json_file)


def obtain_aspects(question_generation_key='diff'):
    save_path = os.path.join(
        question_dir, model_name, question_generation_key, 'diff.json'
    )
    with open(save_path, "r", encoding='utf-8') as json_file:
        diff_dict = json.load(json_file)
    diff_list = diff_dict['diff_list']

    aspect_detail_dict = dict()

    for diff in diff_list:
        print(diff)
        aspect_detail_list = extract_questions(diff)
        for aspect_detail in aspect_detail_list:

            if ':' in aspect_detail:
                aspect, detail = aspect_detail.split(':', maxsplit=1)
            else:
                aspect = aspect_detail.split(':', maxsplit=1)[0]
                detail = ''
            aspect = aspect.lower()
            if aspect not in aspect_detail_dict:
                aspect_detail_dict[aspect] = {'count': 0, 'detail_list': []}
            aspect_detail_dict[aspect]['count'] += 1
            if detail:
                aspect_detail_dict[aspect]['detail_list'].append(detail.strip())

    aspect_detail_dict = dict(sorted(aspect_detail_dict.items(), key=lambda item: item[1]['count'], reverse=True))

    save_path = os.path.join(
        question_dir, model_name, question_generation_key, 'aspect.json'
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding='utf-8') as json_file:
        json.dump(aspect_detail_dict, json_file)


def process_detail(detail):
    return detail.replace('Text 1', 'Text').replace('Text 2', 'Text')


def generate_aspect_instruction(question_generation_key='diff', top_k_aspects=10):
    save_path = os.path.join(
        question_dir, model_name, question_generation_key, 'aspect.json'
    )
    with open(save_path, "r", encoding='utf-8') as json_file:
        aspects_count_dict = json.load(json_file)

    aspect_list = [x for x in list(aspects_count_dict.keys())[:top_k_aspects]]
    instruction_list = []

    prefix_context = model.user_prefix
    train_dataset = [data for data in dataloaders.load_dataset_easy(train_data_id)]
    train_dataset = add_prefix_context_dataset(train_dataset, prefix_context)

    instruction_title = 'Generate an instruction to identify the speaker\'s {}. {} The instructions should request the inclusion of reasoning steps, followed by a conclusion drawn from these steps. Output the instruction only.'
    detail_prompt_title = 'Extract the values of {} mentioned in the above text using one sentence. Start with "For example, the {} could be"'

    for aspect in aspect_list:
        detail_prompt = model.user_prefix
        for detail in aspects_count_dict[aspect]['detail_list'][:10]:
            detail_prompt += process_detail(detail) + '\n'
        detail_prompt += '\n' + detail_prompt_title.format(aspect, aspect) + model.user_assistant_infix
        print(detail_prompt)
        detail = model.generate(detail_prompt)
        print(detail)

        instruction = instruction_title.format(aspect, detail) + model.user_assistant_infix
        print(instruction)
        postfix_content_dict, postfix_context = search_content(instruction, train_dataset, truncate_max_sequence=False)
        postfix_content = postfix_content_dict['text']
        postfix_content = extract_instruction(postfix_content)
        print(postfix_content)
        instruction_list.append(postfix_content)

    instruction_list = list(set(instruction_list))

    save_path = os.path.join(
        question_dir, model_name, question_generation_key,
        'aspect_instruction.json'
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding='utf-8') as json_file:
        json.dump(instruction_list, json_file)


def extend_instruction(postfix_instruction, train_dataset):
    q_extend = model.lm_search(train_dataset, token_length=400,
                               postfix_instruction=postfix_instruction,
                               stop_criterion='eos', stop_by_score=True, metric='ppl_score+acc',
                               top_p=0.9, regularization_label=0)
    return postfix_instruction + q_extend


def extract_instruction(postfix_content):
    instruction_prefix_list = ["instruction:", "instructions:"]
    for instruction_prefix in instruction_prefix_list:
        if instruction_prefix in postfix_content:
            postfix_content = postfix_content.split(instruction_prefix, maxsplit=1)[-1]
            postfix_content = postfix_content.strip()
            break
    else:
        instruction_prefix = ':\n\n"'
        if instruction_prefix in postfix_content:
            postfix_content = postfix_content.split(instruction_prefix, maxsplit=1)[-1]
            postfix_content = postfix_content[:-1]
            postfix_content = postfix_content.strip()
    if postfix_content[0] == '"' and postfix_content[-1] == '"':
        postfix_content = postfix_content[1:-1]
    return postfix_content


def generate_meta_instruction(postfix_instruction_dict):
    prefix_context = model.user_prefix
    train_dataset = [data for data in dataloaders.load_dataset_easy(train_data_id)]
    train_dataset = add_prefix_context_dataset(train_dataset, prefix_context)

    meta_instruction_dict = dict()

    for key, q in postfix_instruction_dict.items():
        postfix_instruction = '\n' + q + model.user_assistant_infix
        print(postfix_instruction)
        postfix_content_dict, postfix_context = search_content(postfix_instruction, train_dataset,
                                                               truncate_max_sequence=False)

        postfix_content = postfix_content_dict['text']
        postfix_content = extract_instruction(postfix_content)
        print(postfix_content)
        meta_instruction_key = get_question_key(postfix_content)
        meta_instruction_dict[meta_instruction_key] = postfix_content
    return meta_instruction_dict


def instruction_train(instruction_dict_name='simple_instruction', extend=True, use_meta_instruction=False):
    postfix_instruction_dict = load_human_defined_contexts_dict(instruction_dict_name)

    print(postfix_instruction_dict)

    if use_meta_instruction:
        postfix_instruction_dict = generate_meta_instruction(postfix_instruction_dict)

    instruction_train_single_dict(postfix_instruction_dict, prefix_context=model.user_prefix,
                                  save_dir=result_instruction_dir, extend=extend, show_top_k=10)


def question_train(question_generation_key, extend=False):
    question_dict = load_question_dict(question_generation_key)
    # question_dict = dict(reversed(question_dict.items()))
    print(question_dict)

    postfix_context_dict_list = instruction_train_single_dict(
        question_dict, prefix_context=model.user_prefix, save_dir=result_instruction_dir, extend=extend, show_top_k=10
    )
    save_dir = os.path.join(result_dir, 'top_diff_instructions', model_name, question_generation_key)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'question_dict.json'), 'w', encoding='utf-8') as f:
        json.dump(postfix_context_dict_list, f)


def search_content(postfix_instruction, train_dataset, truncate_max_sequence=True):
    postfix_content_dict = model.lm_search(
        train_dataset, token_length=1000, postfix_instruction=postfix_instruction,
        return_dict=True, stop_criterion='eos', stop_by_score=False,
        metric='ppl_score+acc', top_p=0.9, regularization_label=0, truncate_max_sequence=truncate_max_sequence
    )
    postfix_context = postfix_instruction + postfix_content_dict['text']
    print(postfix_context)
    print('ppl_score: ', postfix_content_dict['ppl_score'])
    print('ppl_diff: ', postfix_content_dict['ppl_diff'])
    print('acc: ', postfix_content_dict['acc'])
    print('auc: ', postfix_content_dict['auc'])
    return postfix_content_dict, postfix_context


def load_trained_contexts(save_dir, save_tag):
    save_dir = os.path.join(save_dir, model_name, save_tag)
    if not os.path.isdir(save_dir):
        return None, None
    with open(os.path.join(save_dir, 'postfix_context.txt'), 'r', encoding='utf-8') as f:
        postfix_context = f.read()
    with open(os.path.join(save_dir, 'prefix_context.txt'), 'r', encoding='utf-8') as f:
        prefix_context = f.read()
    return prefix_context, postfix_context


def instruction_train_single_dict(postfix_instruction_dict, prefix_context, save_dir, extend, show_top_k=1):
    train_dataset = [data for data in dataloaders.load_dataset_easy(train_data_id)]
    train_dataset = add_prefix_context_dataset(train_dataset, prefix_context)

    postfix_context_dict_list = []
    for key, q in postfix_instruction_dict.items():

        _, postfix_context = load_trained_contexts(save_dir, key)

        if postfix_context is None:
            postfix_instruction = '\n' + q + model.user_assistant_infix
            if extend:
                postfix_instruction = extend_instruction('\n' + q, train_dataset) + model.user_assistant_infix
            print(postfix_instruction)
            postfix_content_dict, postfix_context = search_content(postfix_instruction, train_dataset)
            show_single(postfix_context, prefix_context, save_dir=save_dir, save_tag=key,
                        score_list=postfix_content_dict['score_list'])
        else:
            postfix_content_dict = model.cal_all_metrics(train_dataset, postfix_context)

        postfix_context_dict = {
            'key': key,
            'question': q,
            'postfix_context': postfix_context,
            'score': postfix_content_dict['ppl_score']
        }
        postfix_context_dict_list.append(postfix_context_dict)
    postfix_context_dict_list = sorted(postfix_context_dict_list, key=lambda x: x['score'], reverse=True)
    print(postfix_context_dict_list)
    for top_k in range(show_top_k):
        print("Best {} round top {}: ".format(0, top_k + 1), postfix_context_dict_list[top_k]['postfix_context'])
        print('score: ', postfix_context_dict_list[top_k]['score'])
        show_single(postfix_context_dict_list[top_k]['postfix_context'], prefix_context)
    return postfix_context_dict_list


def main():
    # Example human defined instruction
    instruction = "Discuss anything notable in the above text. Include as much detail as possible."
    standard_train(postfix_instruction=instruction)

    # Difference-based instruction
    question_generation_key = 'diff'
    print(question_generation_key)
    generate_diff(question_generation_key=question_generation_key)
    obtain_aspects(question_generation_key=question_generation_key)
    generate_aspect_instruction(question_generation_key=question_generation_key)
    question_train(question_generation_key=question_generation_key)

    # instruction_dict_name = 'human_instruction'
    # instruction_dict_name = 'common_instruction'
    instruction_dict_name = 'final_instruction'
    instruction_train(instruction_dict_name, extend=False)

    # instruction_dict_name = 'meta_instruction'
    # instruction_train(instruction_dict_name, extend=False, use_meta_instruction=True)


if __name__ == '__main__':
    main()
