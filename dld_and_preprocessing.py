# STEP 1: download all necessary data from lccpu8
# STEP 2: data preprocess of '%stats' tables
import csv
import datetime
import time

path_pre = '/Users/xuyujie/Desktop/'

def prepro_video_load_first(course_id):
    path_in = path_pre + course_id + '/video_load.csv'
    path_out = path_pre + course_id + '/video_load_o.csv'
    f = open(path_in, 'r')
    data = f.readlines()
    with open(path_out, 'wb') as fo:
        wr = csv.writer(fo, quoting=csv.QUOTE_ALL)
        wr.writerow(
            ['module_id', 'module_video_number', 'user_id', 'view_number', 'video_number', 'session_number',
             'only_load_video',
             'only_load_view', 'straight_through', 'start_stop_num', 'skip_ahead_num', 'backward_num', 'slow_rate_num',
             'time_use', 'finish_video_num', 'start', 'end', 'duration'])
        for i in range(1, len(data)):
            try:
                temp = data[i].strip().split(',')
                temp.append((datetime.datetime.strptime(temp[16], "%Y-%m-%d %H:%M:%S") - datetime.datetime.strptime(
                    temp[15], "%Y-%m-%d %H:%M:%S")).total_seconds())
                wr.writerow(temp)
            except:
                print(i, 'error')
        print('v1 done')
        fo.close()
        return 0

def prepro_video_load_second(course_id):
    print('v2 done')
    return 0

# f1 = open('/Users/xuyujie/Desktop/102_1x_4T2015_emnlp/video_load.csv', 'r')
# data = f1.readlines()
# with open('/Users/xuyujie/Desktop/102_1x_4T2015_emnlp/video_load_o.csv', 'wb') as f1o:
#     wr = csv.writer(f1o, quoting=csv.QUOTE_ALL)
#     wr.writerow(
#         ['module_id', 'module_video_number', 'user_id', 'view_number', 'video_number', 'session_number', 'only_load_video',
#          'only_load_view', 'straight_through', 'start_stop_num', 'skip_ahead_num', 'backward_num', 'slow_rate_num',
#          'time_use', 'finish_video_num', 'start', 'end', 'duration'])
#     for i in range(1, len(data)):
#         try:
#             temp = data[i].strip().split(',')
#             temp.append((datetime.datetime.strptime(temp[16], "%Y-%m-%d %H:%M:%S") - datetime.datetime.strptime(temp[15], "%Y-%m-%d %H:%M:%S")).total_seconds())
#             wr.writerow(temp)
#         except:
#             print(i, 'error')

def prepro_others(course_id):
    path_in = path_pre + course_id + '/others.csv'
    path_out = path_pre + course_id + '/others_o.csv'
    f = open(path_in, 'r')
    data = f.readlines()
    with open(path_out, 'wb') as fo:
        wr = csv.writer(fo, quoting=csv.QUOTE_ALL)
        wr.writerow(['user_id', 'module_name', 'active_days', 'page_view', 'event_std', 'start', 'end', 'browser_ratio',
                     'duration'])
        for i in range(1, len(data)):
            try:
                temp = data[i].strip().split(',')
                temp.append((datetime.datetime.strptime(temp[6], "%Y-%m-%d %H:%M:%S") - datetime.datetime.strptime(temp[5], "%Y-%m-%d %H:%M:%S")).total_seconds())
                wr.writerow(temp)
            except:
                print(i, 'error')
        print('o done')
        fo.close()
        return 0

# f2 = open('/Users/xuyujie/Desktop/102_1x_4T2015_emnlp/others.csv', 'r')
# data2 = f2.readlines()
# with open('/Users/xuyujie/Desktop/102_1x_4T2015_emnlp/others_o.csv', 'wb') as f2o:
#     wr = csv.writer(f2o, quoting=csv.QUOTE_ALL)
#     wr.writerow(['user_id','module_name','active_days','page_view','event_std','start','end','browser_ratio', 'duration'])
#     for i in range(1, len(data2)):
#         try:
#             temp = data2[i].strip().split(',')
#             temp.append((datetime.datetime.strptime(temp[6], "%Y-%m-%d %H:%M:%S") - datetime.datetime.strptime(temp[5], "%Y-%m-%d %H:%M:%S")).total_seconds())
#             wr.writerow(temp)
#         except:
#             print(i, 'error')

def prepro_forum(course_id):
    path_in = path_pre + course_id + '/forum.csv'
    path_out = path_pre + course_id + '/forum_o.csv'
    f = open(path_in, 'r')
    data = f.readlines()
    with open(path_out, 'wb') as fo:
        wr = csv.writer(fo, quoting=csv.QUOTE_ALL)
        wr.writerow(['user_id', 'module_id', 'post_num', 'response_num', 'start_time', 'end_time', 'avg_post_length',
                     'responded_num', 'duration'])
        for i in range(1, len(data)):
            try:
                temp = data[i].strip().split(',')
                temp.append((datetime.datetime.strptime(temp[5], "%Y-%m-%d %H:%M:%S") - datetime.datetime.strptime(temp[4], "%Y-%m-%d %H:%M:%S")).total_seconds())
                wr.writerow(temp)
            except:
                print(i, 'error')
        print('f done')
        fo.close()
        return 0

# f4 = open('/Users/xuyujie/Desktop/102_1x_4T2015_emnlp/forum.csv', 'r')
# data4 = f4.readlines()
# with open('/Users/xuyujie/Desktop/102_1x_4T2015_emnlp/forum_o.csv', 'wb') as f4o:
#     wr = csv.writer(f4o, quoting=csv.QUOTE_ALL)
#     wr.writerow(['user_id','module_id','post_num','response_num','start_time','end_time','avg_post_length','responded_num','duration'])
#     for i in range(1, len(data4)):
#         try:
#             temp = data4[i].strip().split(',')
#             temp.append((datetime.datetime.strptime(temp[5], "%Y-%m-%d %H:%M:%S") - datetime.datetime.strptime(temp[4], "%Y-%m-%d %H:%M:%S")).total_seconds())
#             wr.writerow(temp)
#         except:
#             print(i, 'error')

# TODO: assignment needs special preprocess
# Tables for 102_1x_4T2015_assignment:
# 102_1x_4T2015_assignment_stats
# all_max_grade
# HKUSTx_COMP102_1x_4T2015_student_grade
# HKUSTx_COMP102_1x_4T2015_problem_set

import pandas as pd

def prepro_assign_first(course_id):
    path_in = path_pre + course_id + '/assign.csv'
    path_out = path_pre + course_id + '/assign_.csv'
    f = open(path_in, 'r')
    data = f.readlines()
    with open(path_out, 'wb') as fo:
        wr = csv.writer(fo, quoting=csv.QUOTE_ALL)
        wr.writerow(['student_id','module_id','module_name','page_view','distinct_problem_view','distinct_problem_attempt','submission','distinct_correct','avg_solve_time','start','end','grades','duration'])
        for i in range(1, len(data)):
            try:
                temp = data[i].strip().split(',')
                temp.append((datetime.datetime.strptime(temp[10], "%Y-%m-%d %H:%M:%S") - datetime.datetime.strptime(temp[9], "%Y-%m-%d %H:%M:%S")).total_seconds())
                wr.writerow(temp)
            except:
                print(i, 'error')
        print('a1 done')
        fo.close()
        return 0

def get_general_module (course_name, row):
    if course_name == '102_1x_4T2015':
        if row['module_name'][:2] == 'M0':
           return row['module_name']
        if row['module_name'][:2] == 'L0':
           return 'M0'+row['module_name'][2]
        if row['module_name'] == 'Exam':
           return 'M05'
        return 'Error'

    elif course_name == '101x_1T2016' or '102x_1T2016':
        if row['module_name'][:2] == 'M0':
          return row['module_name']
        if row['module_name'][:3] == 'Q01':
          return 'M01'
        if row['module_name'][:3] == 'Q02':
            return 'M02'
        if row['module_name'][:3] == 'Q03':
            return 'M04'
        if row['module_name'][:3] == 'T01':
            return 'M04'
        if row['module_name'][:3] == 'T02':
            return 'M05'
        if row['module_name'][:3] == 'T03':
            return 'M06'
        if row['module_name'] == 'Exam':
            return 'M06'
        return 'Error'

def prepro_assign_second(course_name, course_id, term_id):
    df_assign = pd.read_csv(path_pre+course_name+'/assign_.csv')
    df_problem_set = pd.read_csv(path_pre+course_name+'/problem_set.csv')
    df_all_max_grade = pd.read_csv(path_pre+course_name+'/all_max_grade.csv')
    # df_student_grade = pd.read_csv(path_pre+course_name+'/student_grade.csv')

    df_max_grade = df_all_max_grade[(df_all_max_grade.course_id == course_id) & (df_all_max_grade.term_id == term_id)]
    df_problem_set_count = df_problem_set.groupby(['aggregated_category']).size().reset_index(name='count')

    # new = pd.merge(df_assign, df_student_grade, left_on=['student_id', 'module_name'], right_on=['student_id', 'aggregated_category'])
    new = pd.merge(df_assign, df_max_grade[['problem_type', 'max_grade']], left_on='module_name', right_on='problem_type', how='left')
    new = pd.merge(new, df_problem_set_count, left_on='module_name', right_on='aggregated_category', how='left')

    new['module_general_id'] = new.apply(lambda row: get_general_module(course_name, row), axis=1)

    # Mode 1: not divide ML
    # TODO: here haven't consider calculation of avg_solve_time and duration
    df_sum = new.groupby(['student_id','module_general_id']).sum().reset_index()

    df_sum['avg_prom_page_view'] = df_sum['page_view']/df_sum['distinct_problem_view']
    df_sum['distinct_problem_view_count'] = df_sum['distinct_problem_view']/df_sum['count']
    df_sum['distinct_problem_attempt_count'] = df_sum['distinct_problem_attempt']/df_sum['count']
    df_sum['distinct_problem_attempt_view'] = df_sum['distinct_problem_attempt']/df_sum['distinct_problem_view']
    df_sum['submission_distinct_problem_attempt'] = df_sum['submission']/df_sum['distinct_problem_attempt']
    df_sum['submission_distinct_problem_correct'] = df_sum['submission'] / df_sum['distinct_correct']
    df_sum['distinct_correct_count'] = df_sum['distinct_correct']/df_sum['count']
    df_sum['distinct_correct_attempt'] = df_sum['distinct_correct']/df_sum['distinct_problem_attempt']
    df_sum['grade_ration'] = df_sum['grades']/df_sum['max_grade']

    df_sum_new = df_sum[['student_id','module_general_id','avg_prom_page_view','distinct_problem_view_count','distinct_problem_attempt_count','distinct_problem_attempt_view','submission_distinct_problem_attempt','submission_distinct_problem_correct','distinct_correct_count','distinct_correct_attempt','grade_ration','duration']]

    with open(path_pre+course_name+'/assign_undivided_o.csv', 'wb') as assign_out0:
        df_sum_new.to_csv(assign_out0, encoding='utf-8', index=False)

    print('a2 done 1')

    # Mode 2: divide ML
    new['avg_prom_page_view'] = new['page_view']/new['distinct_problem_view']
    new['distinct_problem_view_count'] = new['distinct_problem_view']/new['count']
    new['distinct_problem_attempt_count'] = new['distinct_problem_attempt']/new['count']
    new['distinct_problem_attempt_view'] = new['distinct_problem_attempt']/new['distinct_problem_view']
    new['submission_distinct_problem_attempt'] = new['submission']/new['distinct_problem_attempt']
    new['submission_distinct_problem_correct'] = new['submission']/new['distinct_correct']
    new['distinct_correct_count'] = new['distinct_correct']/new['count']
    new['distinct_correct_attempt'] = new['distinct_correct']/new['distinct_problem_attempt']
    new['grade_ration'] = new['grades']/new['max_grade']

    feature = new[['student_id','module_general_id','module_name','avg_prom_page_view','distinct_problem_view_count','distinct_problem_attempt_count','distinct_problem_attempt_view','submission_distinct_problem_attempt','submission_distinct_problem_correct','distinct_correct_count','distinct_correct_attempt','grade_ration','duration']]
    # TODO: add new['avg_solve_time']

    if course_name == '102_1x_4T2015':
        df_m = feature.loc[feature['module_name'].str.contains('M')]
        df_l = feature.loc[feature['module_name'].str.contains('L')]
        df_ml = pd.merge(df_m, df_l, how='outer', left_on = ['student_id', 'module_general_id'], right_on = ['student_id', 'module_general_id'])
        df_ml = df_ml.fillna(0)
        with open(path_pre+course_name+'/assign_divided_o.csv', 'wb') as assign_out1:
            df_ml.to_csv(assign_out1, encoding='utf-8', index=False)
        assign_out1.close()
        print('102_1x_4T2015 a2 done 2')

    elif course_name == '101x_1T2016' or '102x_1T2016':
        df_m = feature.loc[feature['module_name'].str.contains('M')]
        df_q = feature.loc[feature['module_name'].str.contains('Q')]
        df_t = feature.loc[feature['module_name'].str.contains('T')]
        df_mq = pd.merge(df_m, df_q, how='outer', left_on=['student_id', 'module_general_id'],
                         right_on=['student_id', 'module_general_id'])
        df_mq = df_mq.fillna(0)
        df_mt = pd.merge(df_m, df_t, how='outer', left_on=['student_id', 'module_general_id'],
                         right_on=['student_id', 'module_general_id'])
        df_mt = df_mt.fillna(0)
        frames = [df_mq, df_mt]
        df_mqt = pd.concat(frames)
        with open(path_pre + course_name + '/assign_divided_o.csv', 'wb') as assign_out1:
            df_mqt.to_csv(assign_out1, encoding='utf-8', index=False)
        assign_out1.close()
        print('101x_1T20161 a2 done 2')

if __name__ == '__main__':
    course1, course_id1, term_id1 = '102_1x_4T2015', 'COMP102.1x', '4T2015'
    prepro_video_load_first(course1)
    prepro_others(course1)
    prepro_forum(course1)
    prepro_assign_first(course1)
    prepro_assign_second(course1, course_id1, term_id1)

    course2, course_id2, term_id2 = '101x_1T2016', 'EBA101x', '1T2016'
    prepro_video_load_first(course2)
    prepro_others(course2)
    prepro_forum(course2)
    prepro_assign_first(course2)
    prepro_assign_second(course2, course_id2, term_id2)

    course3, course_id3, term_id3 = '102x_1T2016', 'EBA102x', '1T2016'
    prepro_video_load_first(course3)
    prepro_others(course3)
    prepro_forum(course3)
    prepro_assign_first(course3)
    prepro_assign_second(course3, course_id3, term_id3)

    # course4, course_id4, term_id4 = '', '', ''
    # course5, course_id5, term_id5 = '', '', ''
    # course6, course_id6, term_id6 = '', '', ''
