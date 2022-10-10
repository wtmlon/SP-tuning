import torch
from transformers import GPT2Tokenizer

from utils import get_verbalization_ids
import torch.nn.functional as F


class PromptEncoder(object):
    def __init__(self, tokenizer, pvp, label_list):
        # Record prompt tokens
        pattern_token_set, pattern_token_indices = set(), []
        # RoBERTa tokenizer is initiated from GPT2Tokenizer,
        # and it tokenizes same words differently in different positions:
        # e.g.  'Hello world!' -> ['Hello', 'Ġworld', '!'];
        #       'Hello', 'world' -> ['Hello'], ['world']
        # So we need to add prefix space to simulate true situations
        kwargs = {'add_prefix_space': True} if isinstance(
            tokenizer, GPT2Tokenizer) else {}
        for idx, part in enumerate(pvp.PATTERN):
            if pvp.BLOCK_FLAG[idx] == 1:
                token_ids = tokenizer.encode(
                    part, add_special_tokens=False, **kwargs)
                pattern_token_set.update(token_ids)
                pattern_token_indices.extend(token_ids)

        # Record label tokens
        label_token_ids = []
        self.labels_len = []
        for label_idx, label in enumerate(label_list):
            verbalizers = pvp.verbalize(label)
            self.labels_len.append(len(verbalizers))
            for verbalizer_idx, verbalizer in enumerate(verbalizers):
                verbalizer_id = get_verbalization_ids(
                    verbalizer, tokenizer, force_single_token=True)
                assert verbalizer_id != tokenizer.unk_token_id, "verbalization was tokenized as <UNK>"
                label_token_ids.append(verbalizer_id)

        assert len(pattern_token_set) < 50 and len(label_token_ids) < 49

        # Convert tokens in manual prompt / label to unused tokens
        # Note that `AlbertTokenizer` or `RobertaTokenizer` doesn't have a `vocab` attribute
        if hasattr(tokenizer, 'vocab') and '[unused0]' in tokenizer.vocab:
            # BERT
            self.pattern_convert = {token_id: tokenizer.vocab['[unused%s]' % idx]
                                    for idx, token_id in enumerate(pattern_token_set)}
            self.label_convert = {token_id: tokenizer.vocab['[unused%s]' % (idx + 50)]
                                  for idx, token_id in enumerate(label_token_ids)}

        else:
            # ALBERT, RoBERTa
            start_idx = tokenizer.vocab_size - 100
            self.pattern_convert = {token_id: start_idx + idx
                                    for idx, token_id in enumerate(pattern_token_set)}
            self.label_convert = {token_id: start_idx + 50 + idx
                                  for idx, token_id in enumerate(label_token_ids)}

        # Convert mlm logits to cls logits
        self.vocab_size = tokenizer.vocab_size
        self.m2c_tensor = torch.tensor( 
            list(self.label_convert.values()), dtype=torch.long)

        # Use lookup tensor to get replace embeddings
        self.lookup_tensor = torch.tensor([self.pattern_convert[origin]
                                           for origin in pattern_token_indices],
                                          dtype=torch.long)

    def init_embed(self, model, random_=False):
        w = model.get_input_embeddings().weight.data
        for origin_id, convert_id in self.pattern_convert.items():
            if random_:
                max_val = w[convert_id].abs().max()
                w[convert_id].uniform_(-max_val, max_val)
            else:
                w[convert_id] = w[origin_id]
        for origin_id, convert_id in self.label_convert.items():
            if random_:
                max_val = w[convert_id].abs().max()
                w[convert_id].uniform_(-max_val, max_val)
            else:
                w[convert_id] = w[origin_id]

    def add_embed_hook(self, model):
        def stop_gradient(_, grad_input, __):
            # grad_input: tuple containing a (vocab_size, hidden_dim) tensor
            return (grad_mask.to(grad_input[0].device) * grad_input[0],)

        # Train certain tokens by multiply gradients with a mask
        trainable_ids = list(self.pattern_convert.values()) + \
            list(self.label_convert.values())
        grad_mask = torch.zeros((self.vocab_size, 1), dtype=torch.float)
        grad_mask[trainable_ids, 0] = 1.0

        return model.get_input_embeddings().register_backward_hook(stop_gradient)
    
    def add_reverse_hook(self, model):
        def stop_gradient(_, grad_input, __):
            # grad_input: tuple containing a (vocab_size, hidden_dim) tensor
            return (grad_mask.to(grad_input[0].device) * grad_input[0],)

        # Train certain tokens by multiply gradients with a mask
        trainable_ids = list(self.pattern_convert.values()) + \
            list(self.label_convert.values())
        grad_mask = torch.ones((self.vocab_size, 1), dtype=torch.float)
        grad_mask[trainable_ids, 0] = 0.0

        return model.get_input_embeddings().register_backward_hook(stop_gradient)

    def get_replace_embeds(self, word_embeddings):
        return word_embeddings(self.lookup_tensor.to(word_embeddings.weight.device))

    def convert_mlm_logits_to_cls_logits(self, mlm_labels, logits):
        labels_logits = torch.index_select(logits[mlm_labels != -1], -1, self.m2c_tensor.to(logits.device))
        cls_logits = torch.zeros(logits.size(0), len(self.labels_len)).to(labels_logits.device)
        for b in range(labels_logits.size(0)):
            begin = 0
            for i in range(len(self.labels_len)):
                cls_logits[b, i] = torch.sum(labels_logits[b, begin : begin + self.labels_len[i]]) / self.labels_len[i]
                begin = self.labels_len[i]

        return cls_logits

    def selective_convert_mlm_logits_to_cls_logits(self, mlm_labels, logits, prompt_embeds, model, bz):
        labels_logits = torch.index_select(logits[mlm_labels > 0], -1, self.m2c_tensor.to(logits.device))
        selective_idx = []
        labels_embeds = []
        begin_list = []
        begin = 0
        for i in range(len(self.labels_len)):
            labels_embeds.append(model.get_input_embeddings().weight[self.m2c_tensor[begin:begin+self.labels_len[i]]].unsqueeze(0))   #dont add .data that will detach graph
            begin_list.append(begin)
            begin = self.labels_len[i]

        #for convert_id in self.m2c_tensor:
        #    labels_embeds.append(model.get_input_embeddings().weight[convert_id].unsqueeze(0))   #dont add .data that will detach graph
        #labels_embeds = torch.cat(labels_embeds).unsqueeze(0).unsqueeze(3).repeat(labels_logits.size(0), 1, 1, prompt_embeds.size(1), 1)   # (bz, cls, cls_cnt, prompt_len, hidden)

        #sim_score = F.cosine_similarity(labels_embeds, prompt_embeds.unsqueeze(1).unsqueeze(1).repeat(1, labels_embeds.size(1), labels_embeds.size(2), 1, 1), dim=4).mean(dim=3).max(2).indices

        #sim_score = sim_score + torch.tensor(begin_list).to(sim_score.device)

        highest_score_idx = []
        for i in range(len(self.labels_len)):
            highest_score_idx.append(labels_logits[:, begin_list[i]:begin_list[i]+self.labels_len[i]].max(1).indices.unsqueeze(0))

        highest_score_idx = torch.cat(highest_score_idx).permute(1,0).to(labels_logits.device) + torch.tensor(begin_list).to(labels_logits.device)

        # ensemble
        m = labels_logits.size(0) // bz
        labels_logit = []
        tmp = []
        for i in range(mlm_labels.size(0)):
            tmp.append([i + j*bz for j in range(m)])
            labels_logit.append(labels_logits[i, highest_score_idx[i]].unsqueeze(0))

        labels_logits = torch.cat(labels_logit)
        tmp = torch.tensor(tmp).to(labels_logits.device)

        ensembles = []
        for i in range(bz):
            ensembles.append(labels_logits[tmp[i]].mean(dim=0).unsqueeze(0))

        labels_logits = torch.cat(ensembles)

        #sim_score = []
        #for b in range(labels_logits.size(0)):
        #    batch_score = []
        #    for embeds in labels_embeds:
        #        batch_score.append(F.cosine_similarity(prompt_embeds[b], embeds.repeat(prompt_embeds.size(1), 1)))

        #    begin = 0
        #    for i in range(len(self.labels_len)):
        #        cls_logits[b, i] = torch.sum(labels_logits[b, begin : begin + self.labels_len[i]]) / self.labels_len[i]
        #        begin = self.labels_len[i]

        #    sim—score.append(batch_score)

        return labels_logits
