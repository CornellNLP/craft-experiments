import json
import os
import string
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
import yaml
from flask import Flask, request, render_template

from convo_wizard.data_processors.tokenizers.convo_tokenizer_v2 import ConvoTokenizer
from convo_wizard.data_processors.tokenizers.utils import generate_from_input_ids_batch
from convo_wizard.models.convo_wizard import ConvoWizard
from convo_wizard.utils.utils import set_seed
from convo_wizard.utils.visualizer import ConvoWizardAttentionVisualizer

app = Flask(__name__)


def load_resources(config_path, tokenizer_filepath, pretrained_checkpoint_path=None, pretrained_model_path=None):
    with open(config_path) as fp:
        config = yaml.safe_load(fp)
    config['general']['device'] = 'cpu'

    tokenizer = ConvoTokenizer.load(tokenizer_filepath)
    model = ConvoWizard(vocab_size=len(tokenizer), padding_idx=tokenizer.pad_token_id,
                        cls_or_sep_token_idx=tokenizer.sep_token_id, device=torch.device('cpu'),
                        **config['transformer']['args'])
    if pretrained_checkpoint_path is not None:
        checkpoint = torch.load(pretrained_checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    elif pretrained_model_path is not None:
        model.from_pretrained(model_path=pretrained_model_path)
    model.eval()
    assert not model.training

    return ConvoWizardAttentionVisualizer(convo_wizard=model, pretrained_tokenizer=tokenizer)


def longest_repeating(_string):
    _string = _string.translate(str.maketrans('', '', string.punctuation)).split(' ')
    maximum = count = 0
    current = ''
    for _tok in _string:
        if _tok == current:
            count += 1
        else:
            count = 1
            current = _tok
        maximum = max(count, maximum)
    return maximum


def get_awry_proba_from_generator(_input_ids, pred_temperature=1.0, device=torch.device('cpu'), prompt_prepended=True):
    awry_label_tok_id = attention_visualizer._tokenizer.vocab['awry_label']
    assert not attention_visualizer._model.training
    if prompt_prepended:
        _input_ids = _input_ids[2:]
    input_ids = torch.hstack(
        (_input_ids, torch.tensor(attention_visualizer._tokenizer.encode('[SEP] >>')[:-1]))).unsqueeze(0)
    input_ids = input_ids.to(device)
    tokenized_convo_new = generate_from_input_ids_batch(input_ids, device=device)
    lm_output, _ = attention_visualizer._model(input_ids=input_ids, position_ids=tokenized_convo_new['position_ids'],
                                               token_type_ids=tokenized_convo_new['token_type_ids'],
                                               attention_mask=tokenized_convo_new['attention_mask'],
                                               make_predictions=False)
    lm_output = lm_output[:, -1, :] / pred_temperature
    probs = F.softmax(lm_output, dim=-1)
    awry_proba = probs[0][awry_label_tok_id].item()
    return awry_proba


def autocomplete(input_ids_, max_num_tokens=50, gen_temperature=1.0, top_k=10, num_samples=10, max_num_utts=1,
                 pred_temperature=1.0, shift_tol=0.1, device=torch.device('cpu'), prompt_prepended=True):
    input_ids = input_ids_.expand(num_samples, -1)
    init_length, init_prompt = input_ids.shape[-1], attention_visualizer._tokenizer.decode(input_ids_)
    num_utts_generated = 0

    label_tok_id = attention_visualizer._tokenizer.vocab['>>']
    prompt_tok_id = attention_visualizer._tokenizer.vocab['<<']
    sep_tok_id = attention_visualizer._tokenizer.vocab['[SEP]']
    for _ in range(max_num_tokens):
        input_ids = input_ids.to(device)
        tokenized_convo_new = generate_from_input_ids_batch(input_ids, device=device)
        lm_output, _ = attention_visualizer._model(input_ids=input_ids,
                                                   position_ids=tokenized_convo_new['position_ids'],
                                                   token_type_ids=tokenized_convo_new['token_type_ids'],
                                                   attention_mask=tokenized_convo_new['attention_mask'],
                                                   make_predictions=False)
        lm_output = lm_output[:, -1, :] / gen_temperature

        if top_k is not None:
            top_k_values, top_k_idxs = torch.topk(lm_output, k=top_k, dim=-1, largest=True, sorted=True)
            lm_output[lm_output < top_k_values[:, [-1]]] = -torch.inf
        probs = F.softmax(lm_output, dim=-1)
        if num_samples > 1:
            next_input_id = torch.multinomial(probs, num_samples=1)
        else:
            _, next_input_id = torch.topk(probs, k=1, dim=-1)
        input_ids = torch.cat((input_ids, next_input_id.to(device)), dim=1)
        if num_samples == 1:
            if next_input_id == sep_tok_id:
                num_utts_generated += 1
            if num_utts_generated >= max_num_utts:
                break

    _input_ids = input_ids[0].cpu().squeeze()
    before_gen_proba = round(get_awry_proba_from_generator(_input_ids[:init_length], pred_temperature=pred_temperature,
                                                           device=device, prompt_prepended=prompt_prepended), 3)
    for idx in range(num_samples):
        _input_ids = input_ids[idx].cpu().squeeze()
        if label_tok_id in _input_ids or prompt_tok_id in _input_ids[2:]:
            continue
        sep_pos = torch.where(_input_ids[init_length:] == sep_tok_id, 1, 0).nonzero()
        gen = _input_ids[init_length:] if len(sep_pos) == 0 else \
            _input_ids[init_length: init_length + sep_pos[0][0].item()]
        decoded_gen = attention_visualizer._tokenizer.decode(gen)
        if '[ deleted ]' in decoded_gen:
            continue
        num_max_repeated_toks = longest_repeating(decoded_gen)
        if num_max_repeated_toks >= 3:
            continue

        after_gen_proba = round(get_awry_proba_from_generator(_input_ids, pred_temperature=pred_temperature,
                                                              device=device, prompt_prepended=prompt_prepended), 3)
        shift = round(before_gen_proba - after_gen_proba, 3)
        if shift > 0 and not (shift < -shift_tol) and after_gen_proba >= 0.005:
            return decoded_gen


@app.route('/')
def home():
    # Note: if you get access denied, flush socket pools at: chrome://net-internals/#sockets.
    return render_template('home.html')


@app.route('/complete', methods=['POST'])
def complete():
    input_convo = request.form.get('convo')
    ignore_punct = True if request.form.get('ignore_punct') == 'true' else False
    temp = float(request.form.get('temp'))
    prompt = 'calm_prompt << '

    input_convo = list(map(str.strip, input_convo.split('[SEP]')))
    input_convo = [prompt + input_convo[0]] + input_convo[1:] if len(input_convo) > 1 else [prompt + input_convo[0]]
    if ignore_punct:
        input_convo = attention_visualizer._remove_punct(input_convo=input_convo)
    convo_input_ids = attention_visualizer._get_input_ids(input_convo=input_convo, append_label_prompt=False).squeeze()
    return autocomplete(input_ids_=convo_input_ids[:-1], max_num_tokens=100, gen_temperature=temp)


@app.route('/visualize', methods=['POST'])
def visualize():
    input_convo = request.form.get('convo')
    ignore_punct = True if request.form.get('ignore_punct') == 'true' else False
    show_all_attn = True if request.form.get('show_all_attn') == 'true' else False
    use_saliency = True if request.form.get('use_saliency') == 'true' else False

    set_seed(42)
    input_convo = list(map(str.strip, input_convo.split('[SEP]')))
    if not use_saliency:
        awry_proba, calm_proba, input_tokens, attention_scores = \
            attention_visualizer.visualize(input_convo=input_convo, get_intermediates=True, ignore_punct=ignore_punct)
        attention_scores = attention_scores * 7
    else:
        awry_proba, calm_proba, input_tokens, attention_scores = \
            attention_visualizer.saliency(input_convo=input_convo, get_intermediates=True, ignore_punct=ignore_punct)
        attention_scores = attention_scores * 10
    attention_scores = attention_scores.numpy().tolist()
    print(f"awry proba: {awry_proba}, calm proba: {calm_proba}")
    if not show_all_attn and '[SEP]' in input_tokens:
        last_utt_start_idx = len(input_tokens) - input_tokens[::-1].index('[SEP]')
        attention_scores = [0.0] * len(attention_scores[:last_utt_start_idx]) + attention_scores[last_utt_start_idx:]
    return json.dumps({
        'tokens': input_tokens,
        'attention_scores': attention_scores,
        'awry_proba': awry_proba,
        'calm_proba': calm_proba,
    })


if __name__ == '__main__':
    parser = ArgumentParser(description='visualize attention from the trained model')
    parser.add_argument('--config_path', type=str, help='path to config file')
    parser.add_argument('--tokenizer_path', type=str, help='path to the pretrained tokenizer', default=os.getcwd())
    parser.add_argument('--pretrained_model_path', type=str, help='path to the pretrained model', default=None)
    parser.add_argument('--pretrained_checkpoint_path', type=str, help='path to the pretrained model checkpoint',
                        default=None)
    parser.add_argument('--port', type=int, help='port number to run the app', default=5000)

    args = parser.parse_args()

    attention_visualizer = load_resources(config_path=args.config_path, tokenizer_filepath=args.tokenizer_path,
                                          pretrained_checkpoint_path=args.pretrained_checkpoint_path,
                                          pretrained_model_path=args.pretrained_model_path)
    app.run(port=args.port)
