import os
import pressure
import zones
import baseline
import pandas as pd
import numpy as np


def extract(directory):
    dataset = pd.DataFrame({'Name File':[],'Baseline':[]})
    dataCount = 0
    for label in os.listdir(directory):
        for subLabel in os.listdir(os.path.join(directory, label)):
            for imgFile in os.listdir(os.path.join(directory, label, subLabel)):
                if imgFile.endswith(".jpg") or imgFile.endswith(".jpeg") or imgFile.endswith(".png"):
                    file_name = directory+'/'+label+'/'+subLabel+'/'+imgFile

                    features = [file_name]
                    features += baseline.extract(file_name)
                    #features += pressure.extract(file_name)
                    #features += zones.extract(file_name)

                    new_data = {'Name File':imgFile,'Baseline':features[1]}

                    dataset = dataset._append(new_data, ignore_index=True)

                    dataCount += 1
                    print(dataCount, imgFile,"Done")
                else:
                    continue
                    
   
     
    return dataset
'''
tes = extract('dataset_image1/')
print(tes)
tes.to_csv('baseline_test_data1.csv', index=False)

df = pd.read_csv('baseline_test_data.csv')

dfa = df._append(tes,ignore_index = True)

dfa.to_csv('baseline_test_data2.csv', index = False)
'''