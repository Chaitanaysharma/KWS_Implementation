#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from Integer_EditDistance import editDistDP
import numpy as np
import json
import pickle
class Hmax:
    def __init__(self):
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
            
        self.pickle_in = open("Confusion.Matrix","rb")
        self.Confusion_Matrix = Conf_Matrix
        self.pickle_in = open("Deletion.vector","rb")
        self.Deletion_vector = Deletion_vector
        self.pickle_in = open("Character_dict","rb")
        self.char_dict = char_dict
        self.integer_dict = dict(map(reversed,char_dict.items()))
        self.pickle_in = open("Insertion.vector","rb")
        self.Insertion_vector = Insertion_vector
        self.Insertion_vector["default"] = 0.0
        self.Deletion_vector["default"] = 0.0

        self.current_max = -1.0
        self.argmax_hypothesis = ""
        self.final_op_list =[]

    def getHmax(self,Target, Encoder_output, row, column, prob, Hypothesis):
        #Hmax = argmax(H) P(T|H)P(H)
        
        shape = Encoder_output.shape
        if( column == shape[1]-1 ):
            prob1 = prob*Encoder_output[row][column]
            Hypothesis1 = Hypothesis[:]
            Hypothesis1.append(row)
            dp,lst = editDistDP(Hypothesis1,Target,len(Hypothesis1),len(Target))
            Prob_T_given_H = 1.0
            for operation_description in lst:
                op = operation_description[0]
                source_position = operation_description[1]
                destination_position = operation_description[2]
                

                if(op == 'replace'):
                   
                    Prob_T_given_H = Prob_T_given_H*self.Confusion_Matrix[Target[destination_position]][Hypothesis1[source_position]]

                if(op == 'insert'):
                    key = self.integer_dict[Target[destination_position]]
                    if(self.Insertion_vector.get(key) == None):
                        key = "default"
                    Prob_T_given_H = Prob_T_given_H*self.Insertion_vector[key]

                if(op == 'delete'):
                    key = self.integer_dict[Hypothesis1[source_position]]
                    if(self.Deletion_vector.get(key)== None):
                        key = "default"
                    Prob_T_given_H = Prob_T_given_H*self.Deletion_vector[key]
            
            PyValue = Prob_T_given_H
            Prob_H_given_T = PyValue*prob1

            if(Prob_H_given_T > self.current_max):
                self.current_max = Prob_H_given_T
                self.argmax_hypothesis = Hypothesis1
                self.final_op_list = lst

            return self.argmax_hypothesis,self.final_op_list  

        for i in range(row,shape[0]):
            prob1 = prob*Encoder_output[i][column]
            Hypothesis1 = Hypothesis[:]
            Hypothesis1.append(i)
            dp,lst = editDistDP(Hypothesis1,Target,len(Hypothesis1),len(Target))
            Prob_T_given_H = 1.0
            for operation_description in lst:
                op = operation_description[0]
                source_position = operation_description[1]
                destination_position = operation_description[2]
                

                if(op == 'replace'):
                   
                    Prob_T_given_H = Prob_T_given_H*self.Confusion_Matrix[Target[destination_position]][Hypothesis1[source_position]]

                if(op == 'insert'):
                    key = self.integer_dict[Target[destination_position]]
                    if(self.Insertion_vector.get(key) == None):
                        key = "default"
                    Prob_T_given_H = Prob_T_given_H*self.Insertion_vector[key]

                if(op == 'delete'):
                    key = self.integer_dict[Hypothesis1[source_position]]
                    if(self.Deletion_vector.get(key)== None):
                        key = "default"
                    Prob_T_given_H = Prob_T_given_H*self.Deletion_vector[key]
            
            PyValue = Prob_T_given_H
            Prob_H_given_T = PyValue*prob1

            if(Prob_H_given_T > self.current_max):
                self.current_max = Prob_H_given_T
                self.argmax_hypothesis = Hypothesis1
                self.final_op_list = lst
            self.getHmax(Target,Encoder_output,i,column+1,prob*Encoder_output[i][column],Hypothesis1)
            

        return self.argmax_hypothesis,self.final_op_list,self.current_max

    

