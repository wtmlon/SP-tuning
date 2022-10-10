from transformers import RobertaTokenizer, RobertaModel
import pdb
tokenizer = RobertaTokenizer.from_pretrained('../model/roberta-large')
model = RobertaModel.from_pretrained('../model/roberta-large')
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)

pdb.set_trace()
