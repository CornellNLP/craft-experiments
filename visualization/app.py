import json
import os
from argparse import ArgumentParser

import torch
import yaml
from flask import Flask, request, render_template

from convo_wizard.data_processors.tokenizers.convo_tokenizer_v2 import ConvoTokenizer
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


@app.route('/')
def home():
    # Note: if you get access denied, flush socket pools at: chrome://net-internals/#sockets.
    return render_template('home.html')


@app.route('/visualize', methods=['POST'])
def visualize():
    input_convo = request.form.get("convo")
    ignore_punct = True if request.form.get("ignore_punct") == 'true' else False

    set_seed(42)
    input_convo = list(map(str.strip, input_convo.split('[SEP]')))
    awry_proba, input_tokens, attention_scores = \
        attention_visualizer.visualize(input_convo=input_convo, get_intermediates=True, ignore_punct=ignore_punct)
    print(awry_proba)
    return json.dumps({
        'tokens': input_tokens,
        'attention_scores': attention_scores.numpy().tolist()[0],
        'awry_proba': awry_proba,
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
