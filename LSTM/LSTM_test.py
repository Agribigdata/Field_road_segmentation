
import os
import torch
import torch.nn as nn
import torch.utils.data as Data
from modules.model import single_LSTM
from torch import optim
import numpy as np
from numpy import mean
from sklearn.metrics import classification_report,accuracy_score
import pandas as pd
import warnings
# from dataset_train.data_split import spilt_data
import shutil
warnings.filterwarnings("ignore")
batch_size = 128
step=512

def get_loader(path):
    test_filename=os.listdir(path+"test")
    test=pd.DataFrame(columns=['vin','speed','ap','jp','br','lon_diff','lat_diff', 'dir_diff','tag'])
    for file in test_filename:
        data=pd.read_excel(path+"test/"+file)
        test=test.append(data)
    test_data=[]
    for i in range(len(test['speed'])):
        temp_test =list(test.iloc[i,[3,4,5,6,7,9,10,11]])
        test_data.append(temp_test)
    if len(test_data)%step!=0:
        shortage=len(test_data)%step
        padding=step-shortage
        #print(padding)
        temp_test = [0,0,0,0,0,0,0,0]
        for j in range(padding):
            test_data.append(temp_test)
    test_data= [test_data[i:i+step] for i in range(0,len(test_data),step)]
    #test_data = torch.tensor(np.array(test_data),dtype=torch.float32)
    test_data=np.array(test_data)

    test_tag=[]
    for i in range(len(test['speed'])):
        temp_tag =list(test.iloc[i,[8]])
        test_tag+=temp_tag
    if len(test_tag)%step!=0:
        shortage=len(test_tag)%step
        padding=step-shortage
        #print(padding)
        temp_test = [0]
        for j in range(padding):
            test_tag+=temp_test
    test_tag= [test_tag[i:i+step] for i in range(0,len(test_tag),step)]
    test_tag = np.array(test_tag)

    test_data=torch.tensor(test_data,dtype=torch.float32)
    test_tag=torch.tensor(test_tag)
    test_dataset=Data.TensorDataset(test_data, test_tag)

    test_loader = Data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)
    return test_loader


DEVICE = "cuda:1"
final_indice=[]
final_y=[]

f_pred=[]
f_true=[]
def evaluate(model,loader):
    # global final_indice,final_y
    model.eval()
    val_loss=[]
    with torch.no_grad():
        for batch in loader:
            b_input_ids, b_labels = batch[0].cuda(), batch[1].cuda()  # shape均为[batch, intervals]
            output = model(b_input_ids, b_labels)
            loss, logit = output[0], output[1]
            val_loss.append(loss.cpu().detach().numpy().item())
            _,logits = torch.max(logit, dim=2)
            f_pred.append(np.concatenate(logits.cpu().numpy()))
            f_true.append(np.concatenate(b_labels.cpu().numpy()))
            final_indice.append(np.concatenate(logits.cpu().numpy()))
            final_y.append(np.concatenate(b_labels.cpu().numpy()))
        loss=np.array(val_loss).mean()
        return loss
'''
def pre_true_ground():
    model.load_state_dict(torch.load('save_model/用来预测的model/best_acc_lstm_600_wheat_exchange_220.pth'))
    for file in test_filename:
        print(file)
        test = pd.DataFrame(columns=["lon", "lat", "dir", "vin", "speed", "ap", "jp", "br", "tag", "lon_diff", "lat_diff", "dir_diff"])
        data = pd.read_csv(path + file)
        test = test.append(data)
        test_data = []
        for i in range(len(test['lon'])):
            temp_test = list(test.iloc[i, [3, 4, 5, 6, 7, 9, 10, 11]])
            test_data.append(temp_test)
        if len(test_data) % step != 0:
            shortage = len(test_data) % step
            padding = step - shortage
            # print(padding)
            temp_test = [0, 0, 0, 0, 0, 0, 0, 0]
            for j in range(padding):
                test_data.append(temp_test)
        test_data = [test_data[i:i + step] for i in range(0, len(test_data), step)]
        test_data = np.array(test_data)
        test_tag = []
        for i in range(len(test['lon'])):
            temp_tag = list(test.iloc[i, [8]])
            test_tag += temp_tag
        if len(test_tag) % step != 0:
            shortage = len(test_tag) % step
            padding = step - shortage
            # print(padding)
            temp_test = [0]
            for j in range(padding):
                test_tag += temp_test
        test_tag = [test_tag[i:i + step] for i in range(0, len(test_tag), step)]
        test_tag = np.array(test_tag)
        test_data = torch.tensor(test_data, dtype=torch.float32)
        test_tag = torch.tensor(test_tag)
        test_dataset = Data.TensorDataset(test_data, test_tag)
        test_loader = Data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
        model.eval()
        val_loss = []
        correct_all = []
        final_indice = []
        final_y = []
        with torch.no_grad():
            for batch in test_loader:
                b_input_ids, b_labels = batch[0].cuda(), batch[1].cuda()  # shape均为[batch, intervals]
                output = model(b_input_ids, b_labels)
                loss, logit = output[0], output[1]
                val_loss.append(loss.cpu().detach().numpy().item())
                logits = torch.argmax(logit, dim=2)
                final_indice.append(np.concatenate(logits.cpu().numpy()))
                final_y.append(np.concatenate(b_labels.cpu().numpy()))
            val_loss = np.array(val_loss).mean()
            print("loss:",val_loss)
            y_final=np.concatenate(final_y)
            indice_final=np.concatenate(final_indice)
            print(classification_report(y_final, indice_final,digits=4))
            indice_final=list(indice_final)
            indice_excel=pd.DataFrame(indice_final)
            indice_excel.columns=['tag']
            #file = open(, "w")
            #indice_excel.to_csv('/home/liguangyuan/GCN_system/all_result/小麦/LSTM/'+file,index=False)
'''

if __name__=="__main__":
    kfold=10
    torch.manual_seed(0)
    accuracy_score_list=[]
    test_road_precision = 0
    test_road_recall = 0
    test_road_f1score = 0
    test_field_precision = 0
    test_field_recall = 0
    test_field_f1score = 0
    test_accuracy = 0
    test_macro_precision = 0
    test_macro_recall = 0
    test_macro_f1score = 0
    test_weight_precision = 0
    test_weight_recall = 0
    test_weight_f1score = 0
    lenset=kfold
    for randomstate in range(kfold):
        print(randomstate)
        #10fold data path
        path = 'xxxx'+str(randomstate)+"/"
        # spilt_data(path, randomstate)
        test_loader=get_loader(path)
        model.load_state_dict(torch.load('save_model/'+str(randomstate)+'paddy_single_lstm_8.pth'))
        evaluate(model,test_loader)
        y_final=np.concatenate(final_y)
        indice_final=np.concatenate(final_indice)
        accuracy_score_list.append(classification_report(y_final, indice_final,digits=4,output_dict=True)['accuracy'])
        test_road_precision += classification_report(y_final, indice_final, digits=4, output_dict=True)['0']['precision']
        test_road_recall += classification_report(y_final, indice_final, digits=4, output_dict=True)['0']['recall']
        test_road_f1score += classification_report(y_final, indice_final, digits=4, output_dict=True)['0']['f1-score']
        test_field_precision += classification_report(y_final, indice_final, digits=4, output_dict=True)['1']['precision']
        test_field_recall += classification_report(y_final, indice_final, digits=4, output_dict=True)['1']['recall']
        test_field_f1score += classification_report(y_final, indice_final, digits=4, output_dict=True)['1']['f1-score']
        test_accuracy += classification_report(y_final, indice_final, digits=4, output_dict=True)['accuracy']
        test_macro_precision += classification_report(y_final, indice_final, digits=4, output_dict=True)['macro avg']['precision']
        test_macro_recall += classification_report(y_final, indice_final, digits=4, output_dict=True)['macro avg']['recall']
        test_macro_f1score += classification_report(y_final, indice_final, digits=4, output_dict=True)['macro avg']['f1-score']
        test_weight_precision += classification_report(y_final, indice_final, digits=4, output_dict=True)['weighted avg']['precision']
        test_weight_recall += classification_report(y_final, indice_final, digits=4, output_dict=True)['weighted avg']['recall']
        test_weight_f1score += classification_report(y_final, indice_final, digits=4, output_dict=True)['weighted avg']['f1-score']

        final_indice=[]
        final_y=[]
    print(accuracy_score_list)
    print('               precision    recall    f1score')
    print('        0    ' + str(round(test_road_precision / lenset, 4)) + '      ' + str(
        round(test_road_recall / lenset, 4)) + '    ' + str(round(test_road_f1score / lenset, 4)))
    print('        1    ' + str(round(test_field_precision / lenset, 4)) + '      ' + str(
        round(test_field_recall / lenset, 4)) + '    ' + str(round(test_field_f1score / lenset, 4)))
    print('\n')
    print('    accuracy                          ' + str(round(test_accuracy / lenset, 4)))
    print('   macro avg    ' + str(round(test_macro_precision / lenset, 4)) + '      ' + str(
        round(test_macro_recall / lenset, 4)) + '    ' + str(round(test_macro_f1score / lenset, 4)))
    print('weighted avg    ' + str(round(test_weight_precision / lenset, 4)) + '      ' + str(
        round(test_weight_recall / lenset, 4)) + '    ' + str(round(test_weight_f1score / lenset, 4)))
    #pre_true_ground()












