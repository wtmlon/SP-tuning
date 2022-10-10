from keytotext import pipeline
import pdb

GEN = True

#def gen_text(keyword):


if GEN:
    nlp = pipeline('k2t-base', model='../model/k2t-base')
    #filename = './data/k-shot/SST-2/16-13/keywords.txt'
    #wfilename = './data/k-shot/SST-2/16-13/aug_train.tsv'
    #rfilename = './data/k-shot/SST-2/16-13/train.tsv'
    filename = './data/k-shot/trec/16-13/keywords.txt'
    wfilename = './data/k-shot/trec/16-13/aug_train.csv'
    rfilename = './data/k-shot/trec/16-13/train.csv'
    samples = []
    with open(filename, 'r') as f, open(wfilename, 'w') as w, open(rfilename, 'r') as r:
        try:
            while True:
                line = f.readline()
                if line:
                    print ("line=",line.split(','))
                    gen_sam = nlp(line.split(',')[:-1])
                    samples.append(gen_sam)
                else:
                    break
        finally:
               f.close()

        for i, line in enumerate(r.readlines()):
            #if i == 0:
            #    w.write(line)
            #    continue
            #line = line.rstrip().split('\t')
            #text_a = line[0]
            #label = line[1]
            #w.write(text_a + '\t' + label + '\n')
            #w.write(samples[i-1] + '\t' + label + '\n')
            line = line.rstrip()
            text_a = line[2:]
            if not text_a.strip():  # Empty sentence
                continue
            label = line[0]
            w.write(line + '\n')
            w.write(label + ',' + samples[i] + '\n')

    pdb.set_trace()
