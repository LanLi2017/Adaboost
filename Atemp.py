# -*- coding: utf-8 -*-
from numpy import *
import numpy as np
from Amodeltemp import AdaboostClassifier
data=[]
# read the original data [seq_number,x,y,z,label, user]
for id in range(1,16):
   with open('../%d.csv'%id,'rt') as csvfile:
       for row in csvfile:
           row=row.split(',')
           row_data=[float(d) for d in row]
           row_data.append(id)
           data.append(row_data)

# duration --for windows length [label,user_id,duration]
duration_record=[]
last_label=None
last_user_id=None
for my_seq,row in enumerate(data):
    # in this way, row=data[my_seq]
    current_label=int(row[4])
    current_user_id=row[5]
    if current_label != last_label or current_user_id != last_user_id:
        duration_record.append([current_label,current_user_id,1])
        last_label=current_label
        last_user_id=current_user_id
    else:
        duration_record[-1][2]+=1  # the duration part of last record +=1
max_window_size=min(row[2] for row in duration_record) # window
max_step=52 # step


def get_raw():
    raw = data
    return raw


def get_userid():
    for id in range(1, 16):
        with open('../%d.csv' % id, 'rt') as csvfile:
            for row in csvfile:
                row = row.split(',')
                row_data = [float(d) for d in row]
                row_data.append(id)
            return row_data[5]


# 将数据集分为 训练，验证， 测试
def split_raw(raw, train_percent, validate_percent,test_percent):
    # raw=get_raw()
    train=[]
    validate=[]
    test=[]
    # m=len(raw)
    m = 15
    train_set=int(m*train_percent)
    validate_set=int(m*validate_percent)
    test_set=m-train_set-validate_set
    perm=np.random.permutation(range(1, 16))
    train_user_id = perm[:train_set]
    validate_user_id = perm[train_set:train_set+validate_set]
    test_user_id = perm[train_set+validate_set:]
    #
    # for i in range(0,int(train_set)):
    #     if(raw[4]==i):
    #        train.append(raw)
    # for i in range(int(train_set),int(train_set+validate_set)):
    #     if(raw[4]==i):
    #         validate.append(raw)
    # for i in range(int(train_set+validate_set),int(train_set+validate_set+test_set)):
    #     if(raw[4]==i):
    #         test.append(raw)
    #
    for row in raw:
        current_user_id = row[5]
        if current_user_id in train_user_id:
            train.append(row)
        elif current_user_id in validate_user_id:
            validate.append(row)
        elif current_user_id in test_user_id:
            test.append(row)
    return train,validate,test

'''
    从window_X（list）中获取他们的x、y、z的平均值，以及x、y、z的方差
    传入 window_X=[[x,y,z],[x,y,z],...]
    return summary_windows=[x,y,z,xs2,ys2,zs2]
'''
import math
def get_xyzs2(window_X):
    _size=len(window_X)
    summary_windows=[0.0]*6
    xyz_mean=[0.0]*3
    xyz_s2=[0.0]*3
    for each in window_X:
        for i in range(3):
            summary_windows[i]+=each[i]
    for i in range(len(xyz_mean)):
        summary_windows[i]=summary_windows[i]/_size
    #求平方差
    for each in window_X:
        for i in range(3):
            summary_windows[i+3]+=(summary_windows[i]-each[i])*(summary_windows[i]-each[i])
    for i in range(3,6):
        summary_windows[i]=math.sqrt(summary_windows[i])
    return summary_windows


# X是一定窗口大小里面所有的加速度数据向量 [[x0,y0,z0],[],...]
# y是 labels
# label 有1234567，其中246为1
def get_data(raw, window_size, step):
    X = []
    y = []
    # raw=get_raw()
    user_id_set = set((row[5] for row in raw))
    for user_id in user_id_set:
        user_data = [row for row in raw if row[5] == user_id]
        for offset in range(0, len(user_data) - window_size, step):
            window_data = user_data[offset:offset + window_size]
            window_X = []
            window_Y = []
            for row in window_data:
                window_X.append(row[1:4])
                window_Y.append(int(row[4]))
            summary_windowsX=get_xyzs2(window_X)
            if (window_Y.count(2)+window_Y.count(4)+window_Y.count(6))>(len(window_Y)/2):
                summary_windowsY=1
            else:
                summary_windowsY=-1
            # if len(set((row[4] for row in window_data))) == 1:  # if there are only one label
            #     if window_data[0][4] == 4:  # label is walking
            #         window_Y = 1
            #     else:
            #         window_Y = -1
            # else:
            #     window_Y = -1
            X.append(summary_windowsX)
            y.append(summary_windowsY)
    return X, y
# X是一定窗口大小里面所有的加速度数据向量 [[x0,y0,z0],[],...]
# y是 labels
# label 有 2  4  6  其中4为1
def get_data_second(raw, window_size, step):
    X = []
    y = []
    # raw=get_raw()
    user_id_set = set((row[5] for row in raw))
    for user_id in user_id_set:
        user_data = [row for row in raw if row[5] == user_id]
        for offset in range(0, len(user_data) - window_size, step):
            window_data = user_data[offset:offset + window_size]
            window_X = []
            window_Y = []
            for row in window_data:
                window_X.append(row[1:4])
                window_Y.append(int(row[4]))
            summary_windowsX=get_xyzs2(window_X)
            if window_Y.count(4)>(len(window_Y)/2):
                summary_windowsY=1
            else:
                summary_windowsY=-1
            # if len(set((row[4] for row in window_data))) == 1:  # if there are only one label
            #     if window_data[0][4] == 4:  # label is walking
            #         window_Y = 1
            #     else:
            #         window_Y = -1
            # else:
            #     window_Y = -1
            X.append(summary_windowsX)
            y.append(summary_windowsY)
    return X, y

def claErrorRate(results, test_y):
    len_test=len(test_y)
    error_count=0.0
    label_pred=np.mat(results)
    label_true=np.mat(test_y).transpose()
    arr_res=label_pred-label_true
    for elem in arr_res:
        if np.abs(elem):
            error_count+=1
        #error_count+=np.abs(elem)
    error_rate=error_count/len_test
    return error_rate


def train(train_data):
    clf = AdaboostClassifier(max_iter=50)
    classifier = clf.fit(train_data[0], train_data[1])  # 训练器 输入 train_data_X, train_data_y
    return classifier

'''
    作为2、4、6的分类器，其中2、6是-1，4是1
'''
def train_second(train_data):
    clf = AdaboostClassifier(max_iter=50)
    classifier = clf.fit(train_data[0], train_data[1])  # 训练器 输入 train_data_X, train_data_y
    return classifier

def evaluate(train_raw, test_raw, window_size, step):
    train_data=get_data(train_raw,window_size,step)
    train_data_second=get_data_second(train_raw,window_size,step)
    test_data=get_data(test_raw,window_size,step)
    
    print("====ready to train")
    model=train(train_data)
    print("====ready to train_second")
    model_second=train(train_data_second)
    print("====train_second finished.")
    #
    clf=AdaboostClassifier(max_iter=100)
    predict_result=clf.predict(test_data[0],model)
    #计算1357的错误数，246的放在另外处理。
    error_count=0
    test_raw_246=[] #表示246的数据
    len_test=len(test_data[1])
    test_raw_246=[]
    for i in range(len_test):
        if i ==-1:
            if test_data[i]!=-1:
                error_count+=1
        else:
            test_raw_246.append(test_raw[i])
    test_data_second=get_data_second(test_raw_246,window_size,step)
    predict_result_second=clf.predict(test_data_second[0],model_second)
    print ("predict_result_second:  ",predict_result_second)
    #
    label_pred_second=np.mat(predict_result_second)
    label_true_second=np.mat(test_data_second[1]).transpose()
    arr_res_second=label_pred_second-label_true_second
    for elem in arr_res_second:
        if np.abs(elem):
            error_count+=1
        #error_count+=np.abs(elem)
    error_rate=(error_count*1.0)/len_test
    print("len_test:",len_test)
    print("error_rate:",error_rate)
    return error_rate


def try_parameter( train_raw,validation_raw):
    (window_size,step)=(8,4)
    score=evaluate(train_raw,validation_raw,window_size,step)
    best_score=np.Inf
    for window_size in range(1, max_window_size + 1):
        for step in range(1, max_step + 1):
            score=evaluate(train_raw,validation_raw,window_size,step)
            print('window_size,step,score,best_score', window_size,step,score,best_score)
            if score < best_score:
                best_score=score
                best_windowsize=window_size
                best_step=step
    return best_windowsize,best_step


def main():
    raw=get_raw()
    train_raw, validation_raw, test_raw=split_raw(raw, 0.899,0.001,0.1)  # Holdout method
    #window_size,step=try_parameter(train_raw,validation_raw)
    (window_size,step)=(8,4)
    result=evaluate(train_raw,test_raw,window_size,step)
    print('result: ', result)


# k-fold cross validation
def divide_data_groups(data,group_of_number):
    group_of_data=[[]for _ in range(group_of_number)] #这个变量不重要 ，只要循环次数
    for i in range(len(data)):
        group_of_data[i%group_of_number].append(data[i])


def evaluate1(train_set, test_set):
    model = train(train_set)
    clf = AdaboostClassifier(max_iter=100)
    predict_result = clf.predict(test_set[0], model)
    evaluate_result1 = claErrorRate(predict_result, test_set[1])
    return evaluate_result1


def main1():
    k=3
    group_of_data=[]
    divide_data_groups(data,k)
    for test_group in range(0,k):
        train_set=[]
        test_set=[]
        for i in range(0,k):
            if i!=test_group :
                train_set=train_set+group_of_data[i]
            else :
                test_set=test_set+group_of_data[i]
    result1=evaluate1(train_set,test_set)
    print('result1:', result1)




if __name__ == '__main__':
     main()
    # main1()
