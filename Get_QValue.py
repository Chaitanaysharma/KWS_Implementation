#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pickle
import torch
import torch.nn.functional as F
import json
import numpy as np
from espnet.utils.io_utils import LoadInputsAndTargets
from espnet.utils.training.batchfy import make_batchset
from espnet.asr.pytorch_backend.asr_init import load_trained_model
from espnet.asr.pytorch_backend.asr import CustomConverter
import cmudict
import re
from Hmax import Hmax
import json
import random

def getPsuedoKeyword(target_phone, already_present):
    a = cmudict.dict()
    b = cmudict.words()
    found = False
    for word in b:
        for lst in a[word]:
            for phone in lst:
                if(target_phone == phone and already_present.get(word) == None):
                    already_present[word] = 1
                    return word
                    
                if(re.search(target_phone,phone) and len(target_phone) !=1 and already_present.get(word) == None):
                    already_present[word] =1
                    return word

device = torch.device("cuda")

with open("/home/ram/chaitanay/features/Great/deltafalse/data.json", "r") as f:
    train_json = json.load(f)["utts"]

with open("/home/ram/chaitanay/features/negative_files/deltafalse/data.json", "r") as f:
    test_json = json.load(f)["utts"]

# with open("/root/kws_data/dump/dev/deltafalse/data.json", "r") as f:
#     dev_json = json.load(f)["utts"]

train_data_batches = make_batchset(train_json, 1)
test_data_batches = make_batchset(test_json, 1)
# dev_data_batches = make_batchset(dev_json, 1)

load_tr = LoadInputsAndTargets(
        mode='asr', load_output=True, preprocess_conf=None,
        preprocess_args={'train': True}  # Switch the mode of preprocessing
)

converter = CustomConverter(subsampling_factor=1, dtype=torch.float32)

model, train_args = load_trained_model("/home/ram/chaitanay/model/model.acc.best")
model = model.to(device=device)
model.dec.sampling_probability = 1.0

phone_to_int = dict(zip(train_args.char_list, np.arange(len(train_args.char_list))))
keyword = "G R EY T"
keyword_tokens = torch.tensor([[phone_to_int[phn] for phn in keyword.split(" ")]]).to(device)

encoder_output = 0

# def get_att_score2(att_w_list):
#     atts = [ele.detach().cpu().numpy().flatten() for ele in att_w_list]
#     atts = np.array(atts)
#     sum_att = np.prod(atts, axis=0)
#     att_score = np.trapz(sum_att)
#     return att_score

# def get_att_score(att_w_list):
#     atts = [ele.detach().cpu().numpy().flatten() for ele in att_w_list]
#     atts = np.array(atts)
#     init_att = atts[0]
#     for att in atts[1:]:
#         init_att = np.abs(init_att - att)
#     att_score = np.mean(init_att)
#     return att_score
decoder_output = 0

for row in train_data_batches:
        data_input = [load_tr(row)]
        data = converter(data_input, device)
        encoder_output, hlens, _ = model.enc(data[0], data[1])
        att_loss, acc, _, decoder_output = model.dec(encoder_output, hlens, keyword_tokens)
        break
        
decoder_output = decoder_output.detach().cpu().numpy()
print(decoder_output.shape)


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
        
        
Q_value = dict()
column =0
pickle_in = open("Confusion.Matrix","rb")
Confusion_Matrix = Conf_Matrix
integer_dict = dict(map(reversed,char_dict.items()))
cmu_dict =cmudict.dict()
keywords = keyword.split()
print("Set of keywords = ",keywords)
for ph in keywords:
    phone = ph.upper()
    print("calculating for phone ",phone)
    Q_x = []
    number = 0
    word_dict = dict()
    word_dict["'kay"]=1
    word_dict["kay"] =1
    while(number <= 5 and len(word_dict) != 500):
        print("Current Iteration Number = ",number)
        class_Hm = Hmax()
        print("Getting a Pseudo Word")
        psd_word = getPsuedoKeyword(phone,word_dict)
        print("Pseudo word = ",psd_word)
        psd_word_lst = list()
        for lst in cmu_dict[psd_word]:
            for cmu_phone in lst:
                if(len(cmu_phone) > 2):
                    cmu_phone = cmu_phone[0:2]
                psd_word_lst.append(char_dict[cmu_phone])
            break
        print("Getting Hmax with respect to ",psd_word)
        H_max,op_list,val=class_Hm.getHmax(psd_word_lst,decoder_output,0,0,1.0,[])
        print("Got Hmax !")
        print("Updating Q value ")
        for op in op_list:
            oper = op[0]
            src_pos = op[1]
            dst_pos = op[2]
            if(oper == 'replace' and psd_word_lst[dst_pos] == char_dict[phone] ):
                val = decoder_output[char_dict[phone]][column]*Confusion_Matrix[char_dict[phone]][H_max[src_pos]]
                Q_x.append(val)
                number = number + 1
            elif(oper == 'delete' and H_max[src_pos] == char_dict[phone] ):
                val = Deletion_vector[phone]
                Q_x.append(val)
                number =number + 1
    sm=0 
    for val in Q_x:
        sm = sm + val
        
    if(Q_value.get(char_dict[phone]) == None):
        if(number != 0):
            Q_value[char_dict[phone]] = sm/float(number)
    else:
        if(number != 0):
            Q_value[char_dict[phone]] = (Q_value[char_dict[phone]] + sm/float(number))/2.0
        

print(Q_value)
        
Threshold_value = 1.0

for val in Q_value:
    Threshold_value = Threshold_value*Q_value[val]
        
        #hs_pad is the output of the encoder
#                 ctc_loss = model.ctc(hs_pad, hlens, keyword_tokens)
#                 ctc_loss = float(ctc_loss.cpu().detach().numpy())
#                 att_loss, acc, _, att_w_list = model.dec(hs_pad, hlens, keyword_tokens)
#                 att_score = get_att_score(att_w_list)

#                 att_loss = float(att_loss.cpu().detach().numpy())
#                 f.write(str(ctc_loss) + "," + str(att_loss) + "," + str(acc) + "," + str(att_score) + "," + str(row[0][1]["output"][0]["token"]))
#                 f.write("\n

Tracking_dict = dict()
Tracking_dict[1] = 0
Tracking_dict[0] = 0

#FINDING THRESHOLDS FOR POSITIVE UTTERANCES 
Target_train = []
for phone in keywords:
    integer_index = char_dict[phone]vi 
    Target_train.append(integer_index)
    
    
    
for row in train_data_batches:
        data_input = [load_tr(row)]
        data = converter(data_input, device)
        encoder_output, hlens, _ = model.enc(data[0], data[1])
        att_loss, acc, _, decoder_output = model.dec(encoder_output, hlens, keyword_tokens)
        decoder_output = decoder_output.detach().cpu().numpy()
        class_Hm = Hmax()
        H_max,op_lst,val = class_Hm.getHmax(Target_train,decoder_output,0,0,1.0,[])
        if( val > Threshold_value):
            count = Tracking_dict[1]
            count = count + 1
            Tracking_dict[1] = count
            

        
ic = 0
for row in test_data_batches:
        data_input = [load_tr(row)]
        data = converter(data_input, device)
        index = random.randint(0, 1000)
        new_data = data[0][:,index:index+200,:]
        new_data_1 = torch.tensor([200],device='cuda:0')
        hs_pad, hlens, _ = model.enc(new_data, new_data_1)
        att_loss, acc, _, decoder_output = model.dec( hs_pad, hlens, keyword_tokens )
        decoder_output = decoder_output.detach().cpu().numpy()
        class_Hm = Hmax()
        H_max,op_lst,val = class_Hm.getHmax(Target_train,decoder_output,0,0,1.0,[])
        if( val > Threshold_value):
            count = Tracking_dict[0]
            count = count + 1
            Tracking_dict[0] = count
        if( ic ==300):
            break
            
        ic = ic +1
            
print(Tracking_dict)        
        
        
        

# with open("dev_confidences_ctc_att_small_sp1_as.txt", "a") as f:
#         for row in dev_data_batches:
#                 data_input = [load_tr(row)]
#                 data = converter(data_input, device)
#                 hs_pad, hlens, _ = model.enc(data[0], data[1])
#                 ctc_loss = model.ctc(hs_pad, hlens, keyword_tokens)
#                 ctc_loss = float(ctc_loss.cpu().detach().numpy())
#                 att_loss, acc, _, att_w_list = model.dec(hs_pad, hlens, keyword_tokens)
#                 att_score = get_att_score(att_w_list)

#                 att_loss = float(att_loss.cpu().detach().numpy())
#                 f.write(str(ctc_loss) + "," + str(att_loss) + "," + str(acc) + "," + str(att_score) + "," + str(row[0][1]["output"][0]["token"]))
                
                
#encoder output is in encoder_output



