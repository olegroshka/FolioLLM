import pandas as pd
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


df = pd.read_csv('../../data/classifier.csv')

model_name = 'FINGU-AI/FinguAI-Chat-v1'
m = MultitaskLM(model_name)

texts_label_0 = df[df['label'] == 0]['text'].tolist()
texts_label_1 = df[df['label'] == 1]['text'].tolist()

tokenizer = AutoTokenizer.from_pretrained(model_name)

tokens_label_0 = tokenizer(texts_label_0, return_tensors='pt', padding=True, truncation=True)
tokens_label_1 = tokenizer(texts_label_1, return_tensors='pt', padding=True, truncation=True)

fo = m.encode(**tokens_label_0)
fg = m.encode(**tokens_label_1)
so = embedding_model.encode(texts_label_0)
sg = embedding_model.encode(texts_label_1)

fom, fgm, som, sgm = fo.mean(0), fg.mean(0), so.mean(0), sg.mean(0)
folio_acc = (((fo @ fom) > (fo @ fgm)).sum() + ((fg @ fgm) > (fg @ fom)).sum()) / 281 # 98%
sentr_acc = (((so @ som) > (so @ sgm)).sum() + ((sg @ sgm) > (sg @ som)).sum()) / 281 # 87%
print(folio_acc, sentr_acc)
