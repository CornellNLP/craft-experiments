from itertools import chain

import numpy as np
from tokenizers import Tokenizer
from tokenizers import models, normalizers, pre_tokenizers, trainers, decoders, processors
from tqdm import trange
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerFast


class ConvoTokenizer(object):
    def __init__(self, convo_corpus, special_toks=None, max_vocab_size=50000, lowercase=True,
                 punct_behavior='contiguous', padding_side='right', truncation_side='left'):
        super().__init__()

        if special_toks is None:
            special_toks = {'pad': '[PAD]', 'unk': '[UNK]', 'cls': '[CLS]', 'sep': '[SEP]', 'mask': '[MASK]'}
        for special_tok_name, special_tok in special_toks.items():
            self.__dict__.update({f'{special_tok_name}_tok': special_tok})
        self._tokenizer = self._build_tokenizer(lowercase=lowercase, punct_behavior=punct_behavior)

        trainer = trainers.WordPieceTrainer(vocab_size=max_vocab_size, special_tokens=list(special_toks.values()))
        self._train_tokenizer(convo_corpus, trainer)
        for special_tok_name, special_tok in special_toks.items():
            self.__dict__.update({f'{special_tok_name}_tok_idx': self._tokenizer.token_to_id(special_tok)})

        self._tokenizer.decoder = decoders.WordPiece(prefix="##")
        self._post_processor()

        self.vocab = self._tokenizer.get_vocab()
        self.pretrained_tokenizer = self._get_pretrained_tokenizer(padding_side=padding_side,
                                                                   truncation_side=truncation_side)

    @property
    def tokenizer(self):
        return self.pretrained_tokenizer

    def __len__(self):
        return self._tokenizer.get_vocab_size()

    def _build_tokenizer(self, lowercase=True, punct_behavior='contiguous'):
        tokenizer = Tokenizer(model=models.WordPiece(unk_token=self.unk_tok))

        normalizers_sequence = [normalizers.NFD(), normalizers.StripAccents()]
        if lowercase:
            normalizers_sequence = [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
        tokenizer.normalizer = normalizers.Sequence(normalizers_sequence)

        pretokenizers_sequence = [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation(behavior=punct_behavior)]
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(pretokenizers_sequence)

        return tokenizer

    def _train_tokenizer(self, corpus, trainer, iter_count=100):
        def corpus_iter_fn():
            for idx in trange(0, len(corpus['train']), iter_count):
                flat_convos = [' '.join(utts) for utts in corpus['train'][idx: idx + iter_count]['convo']]
                yield flat_convos

        self._tokenizer.train_from_iterator(corpus_iter_fn(), trainer)

    def _post_processor(self):
        # Reserve position-0 for [PAD] tokens.
        single_sequence = f'{self.cls_tok}:1 $A:1 {self.sep_tok}:1'
        pair_sequences = f'{self.cls_tok}:1 $A:1 {self.sep_tok}:1 {self.cls_tok}:2 $B:2 {self.sep_tok}:2'

        self._tokenizer.post_processor = \
            processors.TemplateProcessing(single=single_sequence, pair=pair_sequences,
                                          special_tokens=[('[CLS]', self._tokenizer.token_to_id('[CLS]')),
                                                          ('[SEP]', self._tokenizer.token_to_id('[SEP]'))])

    def _get_pretrained_tokenizer(self, padding_side, truncation_side):
        pretrained_tokenizer = PreTrainedTokenizerFast(name_or_path='convo-uncased', tokenizer_object=self._tokenizer,
                                                       unk_token=self.unk_tok, sep_token=self.sep_tok,
                                                       pad_token=self.pad_tok, cls_token=self.cls_tok,
                                                       mask_token=self.mask_tok, padding_side=padding_side,
                                                       truncation_side=truncation_side)
        return pretrained_tokenizer

    def encode(self, utt):
        return self._tokenizer.encode(utt)

    def decode(self, tok_ids):
        return self._tokenizer.decode(tok_ids)

    @staticmethod
    def tokenize(pretrained_tokenizer, convo, max_length=None, pad_token_position=0, pad_tok_type_id=0,
                 labels_ignore_idx=-100):
        cls_tok = pretrained_tokenizer.cls_token
        cls_tok_idx = pretrained_tokenizer.cls_token_id
        sep_tok = pretrained_tokenizer.sep_token
        pad_tok_idx = pretrained_tokenizer.pad_token_id

        if type(convo) == list:
            convo = ' '.join([f'{cls_tok} {utt} {sep_tok}' for utt in convo])
            convo = convo[len(cls_tok): -len(sep_tok)].strip()

        if max_length is not None:
            tokenized_convo = pretrained_tokenizer(convo, padding='max_length', max_length=max_length, truncation=True)
        else:
            tokenized_convo = pretrained_tokenizer(convo, padding=False, truncation=False)
        input_ids = np.array(tokenized_convo['input_ids'])

        position_ids = 1 + np.arange(len(input_ids))
        position_ids = np.where(input_ids == pad_tok_idx, pad_token_position, position_ids)

        cls_mask = np.where(input_ids == cls_tok_idx, 0, labels_ignore_idx)
        cls_idxs = np.concatenate((np.where(cls_mask == 0)[0], [len(cls_mask)]))  # https://arxiv.org/pdf/1908.08345.pdf
        segment_ids = [[int(idx % 2 != 0) + 1] * (cls_idxs[idx + 1] - cls_idxs[idx]) for idx in
                       range(len(cls_idxs) - 1)]
        segment_ids = list(chain.from_iterable(segment_ids))
        assert len(segment_ids) == len(cls_mask)
        segment_ids = np.where(input_ids == pad_tok_idx, pad_tok_type_id, np.array(segment_ids))

        # The position IDs relative to the [CLS] token; [PAD] has index (0).
        relative_position_ids = [1 + np.arange(cls_idxs[_ + 1] - cls_idxs[_]) for _ in range(len(cls_idxs) - 1)]
        relative_position_ids = np.array(list(chain.from_iterable(relative_position_ids)))
        relative_position_ids = np.where(input_ids == pad_tok_idx, pad_token_position, relative_position_ids)

        return {'input_ids': input_ids,
                'position_ids': position_ids,
                'relative_position_ids': relative_position_ids,
                'attention_mask': 1 - np.array(tokenized_convo['attention_mask']),  # reverse to indicate [PAD] tokens
                'cls_mask': cls_mask,
                'sep_mask': None,
                'token_type_ids': segment_ids}

    def save(self, filepath):
        self.pretrained_tokenizer.save_pretrained(filepath)

    @staticmethod
    def load(filepath):
        return AutoTokenizer.from_pretrained(filepath)
