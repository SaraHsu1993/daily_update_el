import pandas as pd
# import datetime
# import csv
# f1 = open('/Users/xuyujie/Desktop/102_1x_4T2015_emnlp/video_load.csv', 'r')
# data = f1.readlines()
# f1o = open('/Users/xuyujie/Desktop/102_1x_4T2015_emnlp/video_load_o.csv', 'wb')
# wr = csv.writer(f1o, quoting=csv.QUOTE_ALL)
# wr.writerow(
#     ['module_id', 'module_video_number', 'user_id', 'view_number', 'video_number', 'session_number', 'only_load_video',
#      'only_load_view', 'straight_through', 'start_stop_num', 'skip_ahead_num', 'backward_num', 'slow_rate_num',
#      'time_use', 'finish_video_num', 'start', 'end', 'duration'])
# for i in range(1, len(data)):
#     temp = data[i].strip().split(',')
#     temp.append((datetime.datetime.strptime(temp[16], "%Y-%m-%d %H:%M:%S") - datetime.datetime.strptime(temp[15], "%Y-%m-%d %H:%M:%S")).total_seconds())
#     wr.writerow(temp)

df_video = pd.read_csv('/Users/xuyujie/Desktop/102_1x_4T2015_emnlp/video_load_o.csv')
df_label = pd.read_csv('/Users/xuyujie/Desktop/102_1x_4T2015_emnlp/label.csv')
new4 = pd.merge(df_video, df_label[['user_id', 'max_module_load']], on=['user_id'])

def fxy(x,y):
   if int(x[2])<y: # -1 means not drop
       return -1
   elif int(x[2])>=y: # 1 means drop
       return 1

new4['label'] = new4.apply(lambda x: fxy(x['module_id'], x['max_module_load']), axis=1)

# TODO: calculate and normalize features
new4_middle1 = new4.drop(new4[(new4.module_id == 'M05')].index)

non_student_id = [
    0,
    4449592,
    4449592,
    2361192,
    2879433,
    4556057,
    4556369,
    4532336,
    5826408,
    8421136,
    6617725,
    3988563]

for i in non_student_id:
    new4_middle1 = new4_middle1.drop(new4_middle1[(new4_middle1.user_id == i)].index)

# # Drop Mode 1: delete users who only watched module1 section1 video
new4_middle1.groupby(['user_id'],as_index = False).sum()
# only_section1_id = []
# for j in only_section1_id:
#     new4_middle1 = new4_middle1.drop(new4_middle1[(new4_middle1.user_id == j)].index)
# # Drop Mode 2: delete users whose total watched video is fewer than certain threshold
# fewer_video_id = []
# for k in fewer_video_id:
#     new4_middle1 = new4_middle1.drop(new4_middle1[(new4_middle1.user_id == k)].index)

new4_middle1['avg_view_number_video'] = new4_middle1['view_number']/new4_middle1['video_number']
new4_middle1['avg_video'] = new4_middle1['video_number']/new4_middle1['module_video_number']
new4_middle1['avg_view_number_session'] = new4_middle1['view_number']/new4_middle1['session_number']
new4_middle1['avg_only_load_video'] = new4_middle1['only_load_video']/new4_middle1['video_number']
new4_middle1['avg_only_load_view'] = new4_middle1['only_load_view']/new4_middle1['view_number']
new4_middle1['avg_straight_through'] = new4_middle1['straight_through']/new4_middle1['video_number']
new4_middle1['avg_start_stop_num'] = new4_middle1['start_stop_num']/new4_middle1['video_number']
new4_middle1['avg_skip_ahead_num'] = new4_middle1['skip_ahead_num']/new4_middle1['video_number']
new4_middle1['avg_backward_num'] = new4_middle1['backward_num']/new4_middle1['video_number']
new4_middle1['avg_slow_rate_num'] = new4_middle1['slow_rate_num']/new4_middle1['video_number']
new4_middle1['avg_time_use'] = new4_middle1['time_use']/new4_middle1['video_number']
new4_middle1['opened_finish_video_num'] = new4_middle1['finish_video_num']/new4_middle1['video_number']
new4_middle1['all_finish_video_num'] = new4_middle1['finish_video_num']/new4_middle1['module_video_number']

feature = new4_middle1[['avg_view_number_video', 'avg_video', 'avg_view_number_session', 'avg_only_load_video', 'avg_only_load_view', 'avg_straight_through', 'avg_start_stop_num', 'avg_skip_ahead_num', 'avg_backward_num', 'avg_slow_rate_num', 'avg_time_use', 'opened_finish_video_num', 'all_finish_video_num', 'duration']]
label = new4_middle1[['label']]

import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split

list_mat = feature.fillna(0).values.tolist()
emnlp_mat = np.array(list_mat)
list_label = label['label'].tolist()
emnlp_label = np.array(list_label)

# from sklearn import preprocessing
# X_train, X_test, Y_train, Y_test = train_test_split(emnlp_mat, emnlp_label, test_size=0.2, random_state=0)
# X_train_scaled = preprocessing.scale(X_train)
# X_test_scaled = preprocessing.scale(X_test)
# clf = svm.SVC(kernel='linear', C=1).fit(X_train_scaled, Y_train)
# clf.score(X_test_scaled, Y_test)
# # print('done modelling')
# print(clf.score(X_test_scaled, Y_test))

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler(copy=True, feature_range=(0, 1))
X_minmax = min_max_scaler.fit_transform(emnlp_mat)

# TODO: feature selection
# from sklearn.feature_selection import VarianceThreshold
# sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
# sel.fit_transform(X_train_scaled)

# 5-fold cross validation
from sklearn.model_selection import cross_val_score
clf = svm.SVC(kernel='linear', C=1)
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
accuracy = cross_val_score(clf, X=X_minmax,y=emnlp_label,cv=5,scoring=make_scorer(accuracy_score))
precision = cross_val_score(clf, X=X_minmax,y=emnlp_label,cv=5,scoring=make_scorer(precision_score))
recall = cross_val_score(clf, X=X_minmax,y=emnlp_label,cv=5,scoring=make_scorer(recall_score))
f1 = cross_val_score(clf, X=X_minmax,y=emnlp_label,cv=5,scoring=make_scorer(f1_score))
print('Accuracy', accuracy)
print('Precision', precision)
print('Recall', recall)
print('F1 score', f1)
