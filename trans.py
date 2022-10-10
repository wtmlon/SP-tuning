import translators
import linecache
import os

def mr_trec_cr_subj_trans_data(path):
    with open(path, encoding='utf8') as f:
        tran_file = path + '1'
        if not os.path.exists(tran_file):
            os.system(r"touch {}".format(tran_file))
   
        with open(path + '1', 'w', encoding='utf8') as tf:
            for i, line in enumerate(f.readlines()):
                line = line.rstrip()
                text_a = line[2:]
                if not text_a.strip():  # Empty sentence
                    continue
                tr_a = translators.youdao(query_text=text_a, from_language='en', to_language='zh-CHS', proxy=None)
                trb_a = translators.youdao(query_text=tr_a, from_language='zh-CHS', to_language='en', proxy=None)
                print(trb_a)
                tf.write(trb_a + '\n')

def SST2_trans_data(path):
    with open(path, encoding='utf8') as f:
        tran_file = path + '1'
        if not os.path.exists(tran_file):
            os.system(r"touch {}".format(tran_file))
   
        with open(path + '1', 'w', encoding='utf8') as tf:
            for i, line in enumerate(f.readlines()):
                if i == 0:
                    continue
                line = line.rstrip().split('\t')
                text_a = line[0]
                tr_a = translators.youdao(query_text=text_a, from_language='en', to_language='zh-CHS', proxy=None)
                trb_a = translators.youdao(query_text=tr_a, from_language='zh-CHS', to_language='en', proxy=None)
                print(trb_a)
                tf.write(trb_a + '\n')

def MNLI_trans_data(path):
    with open(path, encoding='utf8') as f:
        tran_file = path + '1'
        if not os.path.exists(tran_file):
            os.system(r"touch {}".format(tran_file))
   
        with open(path + '1', 'w', encoding='utf8') as tf:
            for i, line in enumerate(f.readlines()):
                if i == 0:
                    continue
                line = line.rstrip().split('\t')
                text_a = line[8]
                text_b = line[9]
                tr_a = translators.youdao(query_text=text_a, from_language='en', to_language='zh-CHS', proxy=None)
                tr_b = translators.youdao(query_text=text_b, from_language='en', to_language='zh-CHS', proxy=None)
                trb_a = translators.youdao(query_text=tr_a, from_language='zh-CHS', to_language='en', proxy=None)
                trb_b = translators.youdao(query_text=tr_b, from_language='zh-CHS', to_language='en', proxy=None)
                print(trb_a + '\t' + trb_b)
                tf.write(trb_a + '\t' + trb_b + '\n')

def SNLI_trans_data(path):
    with open(path, encoding='utf8') as f:
        tran_file = path + '1'
        if not os.path.exists(tran_file):
            os.system(r"touch {}".format(tran_file))
   
        with open(path + '1', 'w', encoding='utf8') as tf:
            for i, line in enumerate(f.readlines()):
                if i == 0:
                    continue
                line = line.rstrip().split('\t')
                text_a = line[7]
                text_b = line[8]
                tr_a = translators.youdao(query_text=text_a, from_language='en', to_language='zh-CHS', proxy=None)
                tr_b = translators.youdao(query_text=text_b, from_language='en', to_language='zh-CHS', proxy=None)
                trb_a = translators.youdao(query_text=tr_a, from_language='zh-CHS', to_language='en', proxy=None)
                trb_b = translators.youdao(query_text=tr_b, from_language='zh-CHS', to_language='en', proxy=None)
                print(trb_a + '\t' + trb_b)
                tf.write(trb_a + '\t' + trb_b + '\n')

def QNLI_trans_data(path):
    with open(path, encoding='utf8') as f:
        tran_file = path + '1'
        if not os.path.exists(tran_file):
            os.system(r"touch {}".format(tran_file))
   
        with open(path + '1', 'w', encoding='utf8') as tf:
            for i, line in enumerate(f.readlines()):
                if i == 0:
                    continue
                line = line.rstrip().split('\t')
                text_a = line[1]
                text_b = line[2]
                tr_a = translators.youdao(query_text=text_a, from_language='en', to_language='zh-CHS', proxy=None)
                tr_b = translators.youdao(query_text=text_b, from_language='en', to_language='zh-CHS', proxy=None)
                trb_a = translators.youdao(query_text=tr_a, from_language='zh-CHS', to_language='en', proxy=None)
                trb_b = translators.youdao(query_text=tr_b, from_language='zh-CHS', to_language='en', proxy=None)
                print(trb_a + '\t' + trb_b)
                tf.write(trb_a + '\t' + trb_b + '\n')

def MRPC_QQP_trans_data(path):
    with open(path, encoding='utf8') as f:
        tran_file = path + '1'
        if not os.path.exists(tran_file):
            os.system(r"touch {}".format(tran_file))
   
        with open(path + '1', 'w', encoding='utf8') as tf:
            for i, line in enumerate(f.readlines()):
                if i == 0:
                    continue
                line = line.rstrip().split('\t')
                text_a = line[3]
                text_b = line[4]
                tr_a = translators.youdao(query_text=text_a, from_language='en', to_language='zh-CHS', proxy=None)
                tr_b = translators.youdao(query_text=text_b, from_language='en', to_language='zh-CHS', proxy=None)
                trb_a = translators.youdao(query_text=tr_a, from_language='zh-CHS', to_language='en', proxy=None)
                trb_b = translators.youdao(query_text=tr_b, from_language='zh-CHS', to_language='en', proxy=None)
                print(trb_a + '\t' + trb_b)
                tf.write(trb_a + '\t' + trb_b + '\n')

seed = [13, 21, 42, 87, 100]
for s in seed:
    path_s = os.path.join('./data/k-shot/MNLI/16-{}'.format(s), 'train.tsv')
    MNLI_trans_data(path_s)

for s in seed:
    path_s = os.path.join('./data/k-shot/SNLI/16-{}'.format(s), 'train.tsv')
    SNLI_trans_data(path_s)

for s in seed:
    path_s = os.path.join('./data/k-shot/QNLI/16-{}'.format(s), 'train.tsv')
    QNLI_trans_data(path_s)

for s in seed:
    path_s = os.path.join('./data/k-shot/MRPC/16-{}'.format(s), 'train.tsv')
    MRPC_QQP_trans_data(path_s)

for s in seed:
    path_s = os.path.join('./data/k-shot/QQP/16-{}'.format(s), 'train.tsv')
    MRPC_QQP_trans_data(path_s)

for s in seed:
    path_s = os.path.join('./data/k-shot/SST-2/16-{}'.format(s), 'train.tsv')
    SST2_trans_data(path_s)

for s in seed:
    path_s = os.path.join('./data/k-shot/mr/16-{}'.format(s), 'train.tsv')
    mr_trec_cr_subj_trans_data(path_s)

for s in seed:
    path_s = os.path.join('./data/k-shot/cr/16-{}'.format(s), 'train.tsv')
    mr_trec_cr_subj_trans_data(path_s)

for s in seed:
    path_s = os.path.join('./data/k-shot/trec/16-{}'.format(s), 'train.tsv')
    mr_trec_cr_subj_trans_data(path_s)

for s in seed:
    path_s = os.path.join('./data/k-shot//16-{}'.format(s), 'train.tsv')
    mr_trec_cr_subj_trans_data(path_s)
