#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import json
import pickle

Conf_Matrix = np.zeros((43,43))
Insertion_vector = dict()
Deletion_vector = dict()

model_json = json.load(open("/home/ram/chaitanay/model/model.json"))

lst = model_json[2]['char_list']

char_dict = dict()
index = 0

for ch in lst:
    char = ch.upper()
    char_dict[char] = index
    index = index + 1


f_reference = open("ref.trn")

total_number_dict = dict()

for line in f_reference:
    array = line.split()
    for w in array:
        word = w.upper()
        if(total_number_dict.get(word) != None):
            count = total_number_dict[word]
            count = count + 1
            total_number_dict[word] = count
        else:
            total_number_dict[word] = 1

f_statistics = open("hyp.trn.dtl")

for line in f_statistics:
    array = line.split()
    occurances = int(array[1])
    word1 = array[3].upper()  # word1 => word2
    word2 = array[5].upper()
    prob = occurances/float(total_number_dict[word1])
    Conf_Matrix[char_dict[word1]][char_dict[word2]] = prob

# Conf_Matrix[char_dict["IH"]][char_dict["AH"]]
# with open("Confusion.Matrix",'wb') as cm:
#     pickle.dump(Conf_Matrix,cm)
# with open("Character_dict",'wb') as cd:
#     pickle.dump(char_dict,cd)

f_insertion = open("Insertion")

for line in f_insertion:
    array = line.split()
    occurances = int(array[1])
    word = array[3].upper()
    prob = occurances/float(total_number_dict[word])
    Insertion_vector[word] = prob
    
f_deletion = open("Deletion")
for line in f_deletion:
    array = line.split()
    occurances = int(array[1])
    word = array[3].upper()
    prob = occurances/float(total_number_dict[word])
    Deletion_vector[word] = prob
    



# In[ ]:




