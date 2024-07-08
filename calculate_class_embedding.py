import torch
import os

data_path='test_data/fl-Step3-BFR-e4e'
files = os.listdir(data_path)
cates={}

for file in files:
    cate = file.split('-')[0]
    value = torch.load(data_path+'/'+file)
    if cate not in cates.keys():
        cates[cate]=[value]
    else:
        cates[cate].append(value)

for key in cates.keys():
    cate_embedding = torch.stack(cates[key],dim=0)
    cate_embedding = cate_embedding.mean(dim=0)
    cates[key] = cate_embedding
torch.save(cates,'test_data/fl-class_embeddings.pt')