from operator import le
from statistics import mode
import numpy as np
import json
from nltk_utils import tokenize, stem, bag_of_word
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet



with open('intents.json','r') as f:
    intents = json.load(f)
    #print(intents)


all_words =[]
tags =[]
xy =[]

for intent in intents['intents']:
    tag = intent['tag']
    #print(tag)
    tags.append(tag)
    for pattern in intent['patterns']:
        #print(pattern)
        w = tokenize(pattern)
        #print(w)
        all_words.extend(w)
        xy.append((w,tag))

ignore_words = ['?','!']

# for w in all_words:
#     all_words = stem(w)
#     if w not in ignore_words:
#      #print(all_words)
#      all_words  = sorted(set(all_words))
#      tags = sorted(set(tags))
# # all_words = stem(w)
# # print(all_words)

# # for w in all_words:
# #      w= stem(w)
# #      if w not in ignore_words:
# #         print(all_words)

all_words = [stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

#print(tags)

print(all_words)


X_train = []
y_train =[]

for (pattern_sentence,tag) in xy:
    bag = bag_of_word(pattern_sentence,all_words)
    print(bag)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)


X_train = np.array(X_train)
y_train = np.array(y_train)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples =len(X_train)
        self.x_data = X_train
        self.y_data =y_train
    

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]

    
    def __len__(self):
        return self.n_samples

batch_size =8
hidden_size =8
learning_rate =0.001
num_epoch =1000
output_size = len(tags)
input_size = len(X_train[0])
print(input_size, len(all_words))
print(output_size,tags)

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)
print(train_loader)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)


criterion = nn.CrossEntropyLoss()
optimizer =torch.optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(num_epoch):
    for (words, labels) in train_loader:
        #print(words)
        words =words.to(device)
        #labels =labels.to(device)
        labels = labels.cuda().long()

        outputs = model(words)
        loss =criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print (f'Epoch [{epoch+1}]')
        print (f'Epoch [{epoch+1}/{1000}], Loss: {loss.item():.4f}')



data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
        