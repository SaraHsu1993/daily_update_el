# run for only one course
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


path_pre = '/Users/xuyujie/Desktop/'

# assign_mode = divided, undivided
# user_mode = all, active, passive
# week_mode = one, two
# user_mode_threshold = 1,2,3,...
# video_mode = m1s1, threshold
def file2mat(course_id, assign_mode, user_mode, user_mode_threshold, week_mode, video_mode):
    df_video_load = pd.read_csv(path_pre+course_id+'/video_load_o.csv')
    df_assign_divided = pd.read_csv(path_pre+course_id+'/assign_divided_o.csv')
    df_assign_undivided = pd.read_csv(path_pre+course_id+'/assign_undivided_o.csv')
    df_forum = pd.read_csv(path_pre+course_id+'/forum_o.csv')
    df_other = pd.read_csv(path_pre+course_id+'/others_o.csv')
    df_label = pd.read_csv(path_pre+course_id+'/label.csv')

    if assign_mode == 'divided':
        new1 = pd.merge(df_video_load, df_assign_divided, how='outer', left_on=['module_id', 'user_id'], right_on=['module_general_id', 'student_id'])
    elif assign_mode == 'undivided':
        new1 = pd.merge(df_video_load, df_assign_undivided, how='outer', left_on=['module_id', 'user_id'], right_on=['module_general_id', 'student_id'])

    new1.module_id.fillna(new1.module_general_id, inplace=True)
    new1.user_id.fillna(new1.student_id, inplace=True)

    if user_mode == 'active':
        # TODO: active learner
        new2 = pd.merge(new1, df_forum, left_on=['module_id', 'user_id'], right_on=['module_id', 'user_id'])
        print(user_mode)
        # print(new2.shape)
    elif user_mode == 'all' or 'passive':
        # TODO: passive learner
        new2 = pd.merge(new1, df_forum, how='outer', left_on=['module_id', 'user_id'], right_on=['module_id', 'user_id'])
        print(user_mode)
        # print(new2.shape)

    new3 = pd.merge(new2, df_other, how='left', left_on=['module_id', 'user_id'], right_on=['module_name', 'user_id'])
    new4 = pd.merge(new3, df_label[['user_id', 'max_module_load']], how='left', on=['user_id'])

    new4 = new4.fillna(0)
    new4 = new4[new4['module_id'].str.contains("M0") == True]

    if course_id == '102_1x_4T2015':
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

    elif course_id == '101x_1T2016':
        non_student_id = [
            0,
            6100433,
            8665847,
            8812109,
            3615721,
            652663,
            11,
            9294520,
            9798429,
            961866,
            10378507,
            543339]

    elif course_id == '102x_1T2016':
        non_student_id = [
            0,
            1628821,
            1876953,
            2361192,
            2879433,
            4350243,
            4556369,
            5054267,
            5116254]

    for i in non_student_id:
        new4_middle1 = new4.drop(new4[(new4.user_id == i)].index)

    # if video_mode == 'm1s1':
    #     # TODO: need to get necessary user_id
    #     # Drop Mode 1: delete users who only watched module1 section1 video
    #     only_m1s1 = open(path_pre+course_id+'/out.txt', 'r')
    #     only_m1s1_id = only_m1s1.readlines()
    #     for j in only_m1s1_id:
    #         new4_middle1 = new4_middle1.drop(new4_middle1[(new4_middle1.user_id == int(j.strip()))].index)

    if video_mode =='threshold':
        # Drop Mode 2: delete users whose total watched video is fewer than certain threshold
        count = new4_middle1.groupby(['user_id'])['video_number'].sum().reset_index()
        fewer_video_id = count[count['video_number'] <= user_mode_threshold]['user_id'].values.tolist()
        for k in fewer_video_id:
            new4_middle1 = new4_middle1.drop(new4_middle1[(new4_middle1.user_id == k)].index)

    if user_mode == 'passive':
        # TODO: passive learner
        active_stu = pd.merge(new1, df_forum, left_on=['module_id', 'user_id'], right_on=['module_id', 'user_id'])
        active_stu_id = active_stu['user_id'].values.tolist()
        for l in active_stu_id:
            new4_middle1 = new4_middle1.drop(new4_middle1[(new4_middle1.user_id == l)].index)
    print(user_mode)
    # print(new4_middle1.shape)

    new4_middle1['avg_view_number_video'] = new4_middle1['view_number'] / new4_middle1['video_number']
    new4_middle1['avg_video'] = new4_middle1['video_number'] / new4_middle1['module_video_number']
    new4_middle1['avg_view_number_session'] = new4_middle1['view_number'] / new4_middle1['session_number']
    new4_middle1['avg_only_load_video'] = new4_middle1['only_load_video'] / new4_middle1['video_number']
    new4_middle1['avg_only_load_view'] = new4_middle1['only_load_view'] / new4_middle1['view_number']
    new4_middle1['avg_straight_through'] = new4_middle1['straight_through'] / new4_middle1['video_number']
    new4_middle1['avg_start_stop_num'] = new4_middle1['start_stop_num'] / new4_middle1['video_number']
    new4_middle1['avg_skip_ahead_num'] = new4_middle1['skip_ahead_num'] / new4_middle1['video_number']
    new4_middle1['avg_backward_num'] = new4_middle1['backward_num'] / new4_middle1['video_number']
    new4_middle1['avg_slow_rate_num'] = new4_middle1['slow_rate_num'] / new4_middle1['video_number']
    new4_middle1['avg_time_use'] = new4_middle1['time_use'] / new4_middle1['video_number']
    new4_middle1['opened_finish_video_num'] = new4_middle1['finish_video_num'] / new4_middle1['video_number']
    new4_middle1['all_finish_video_num'] = new4_middle1['finish_video_num'] / new4_middle1['module_video_number']

    new4_middle1['avg_responded_num'] = new4_middle1['responded_num'] / new4_middle1['post_num']

    if week_mode == 'one':
        def f_one_week_label(x, y):
            if int(x[2]) < y:
                return -1
            elif int(x[2]) >= y:
                return 1
        new4_middle1['label'] = new4_middle1.apply(lambda x: f_one_week_label(x['module_id'], x['max_module_load']), axis=1)
        if course_id == '102_1x_4T2015':
            new4_middle1 = new4_middle1.drop(new4_middle1[(new4_middle1.module_id == 'M05')].index)
        elif course_id == '101x_1T2016' or '102x_1T2016':
            new4_middle1 = new4_middle1.drop(new4_middle1[(new4_middle1.module_id == 'M06')].index)

    elif week_mode == 'two':
        if course_id == '102_1x_4T2015':
            df_m01 = new4_middle1.loc[new4_middle1['module_id'].str.contains('M01')]
            df_m02 = new4_middle1.loc[new4_middle1['module_id'].str.contains('M02')]
            df_m03 = new4_middle1.loc[new4_middle1['module_id'].str.contains('M03')]
            df_m04 = new4_middle1.loc[new4_middle1['module_id'].str.contains('M04')]
            # df_m12 = pd.merge(df_m01, df_m02, how='outer', on=['user_id'])
            # df_m12.module_id_x.fillna(df_m12.module_id_y, inplace=True)
            # df_m12.max_module_load_x.fillna(df_m12.max_module_load_y, inplace=True)
            # df_m23 = pd.merge(df_m02, df_m03, how='outer', on=['user_id'])
            # df_m23.module_id_x.fillna(df_m23.module_id_y, inplace=True)
            # df_m23.max_module_load_x.fillna(df_m23.max_module_load_y, inplace=True)
            # df_m34 = pd.merge(df_m03, df_m04, how='outer', on=['user_id'])
            # df_m34.module_id_x.fillna(df_m34.module_id_y, inplace=True)
            # df_m34.max_module_load_x.fillna(df_m34.max_module_load_y, inplace=True)
            # df_frames = [df_m12, df_m23, df_m34]
            # df_two_week_all = pd.concat(df_frames)
            # df_two_week_all = df_two_week_all.fillna(0)
            # print('outer', df_two_week_all.shape)

            ddf_m12 = pd.merge(df_m01, df_m02, on=['user_id'])
            ddf_m12.module_id_x.fillna(ddf_m12.module_id_y, inplace=True)
            ddf_m12.max_module_load_x.fillna(ddf_m12.max_module_load_y, inplace=True)
            ddf_m23 = pd.merge(df_m02, df_m03, on=['user_id'])
            ddf_m23.module_id_x.fillna(ddf_m23.module_id_y, inplace=True)
            ddf_m23.max_module_load_x.fillna(ddf_m23.max_module_load_y, inplace=True)
            ddf_m34 = pd.merge(df_m03, df_m04, on=['user_id'])
            ddf_m34.module_id_x.fillna(ddf_m34.module_id_y, inplace=True)
            ddf_m34.max_module_load_x.fillna(ddf_m34.max_module_load_y, inplace=True)
            ddf_frames = [ddf_m12, ddf_m23, ddf_m34]
            ddf_two_week_all = pd.concat(ddf_frames)
            ddf_two_week_all = ddf_two_week_all.fillna(0)
            print('inner', ddf_two_week_all.shape)

        elif course_id == '101x_1T2016' or '102x_1T2016':
            df_m01 = new4_middle1.loc[new4_middle1['module_id'].str.contains('M01')]
            df_m02 = new4_middle1.loc[new4_middle1['module_id'].str.contains('M02')]
            df_m03 = new4_middle1.loc[new4_middle1['module_id'].str.contains('M03')]
            df_m04 = new4_middle1.loc[new4_middle1['module_id'].str.contains('M04')]
            df_m05 = new4_middle1.loc[new4_middle1['module_id'].str.contains('M05')]
            # df_m12 = pd.merge(df_m01, df_m02, how='outer', on=['user_id'])
            # df_m12.module_id_x.fillna(df_m12.module_id_y, inplace=True)
            # df_m12.max_module_load_x.fillna(df_m12.max_module_load_y, inplace=True)
            # df_m23 = pd.merge(df_m02, df_m03, how='outer', on=['user_id'])
            # df_m23.module_id_x.fillna(df_m23.module_id_y, inplace=True)
            # df_m23.max_module_load_x.fillna(df_m23.max_module_load_y, inplace=True)
            # df_m34 = pd.merge(df_m03, df_m04, how='outer', on=['user_id'])
            # df_m34.module_id_x.fillna(df_m34.module_id_y, inplace=True)
            # df_m34.max_module_load_x.fillna(df_m34.max_module_load_y, inplace=True)
            # df_m45 = pd.merge(df_m04, df_m05, how='outer', on=['user_id'])
            # df_m45.module_id_x.fillna(df_m45.module_id_y, inplace=True)
            # df_m45.max_module_load_x.fillna(df_m45.max_module_load_y, inplace=True)
            # df_frames = [df_m12, df_m23, df_m34, df_m45]
            # df_two_week_all = pd.concat(df_frames)
            # df_two_week_all = df_two_week_all.fillna(0)
            # print('outer', df_two_week_all.shape)

            ddf_m12 = pd.merge(df_m01, df_m02, on=['user_id'])
            ddf_m12.module_id_x.fillna(ddf_m12.module_id_y, inplace=True)
            ddf_m12.max_module_load_x.fillna(ddf_m12.max_module_load_y, inplace=True)
            ddf_m23 = pd.merge(df_m02, df_m03, on=['user_id'])
            ddf_m23.module_id_x.fillna(ddf_m23.module_id_y, inplace=True)
            ddf_m23.max_module_load_x.fillna(ddf_m23.max_module_load_y, inplace=True)
            ddf_m34 = pd.merge(df_m03, df_m04, on=['user_id'])
            ddf_m34.module_id_x.fillna(ddf_m34.module_id_y, inplace=True)
            ddf_m34.max_module_load_x.fillna(ddf_m34.max_module_load_y, inplace=True)
            ddf_m45 = pd.merge(df_m04, df_m05, on=['user_id'])
            ddf_m45.module_id_x.fillna(ddf_m45.module_id_y, inplace=True)
            ddf_m45.max_module_load_x.fillna(ddf_m45.max_module_load_y, inplace=True)
            ddf_frames = [ddf_m12, ddf_m23, ddf_m34, ddf_m45]
            ddf_two_week_all = pd.concat(ddf_frames)
            ddf_two_week_all = ddf_two_week_all.fillna(0)
            print('inner', ddf_two_week_all.shape)

        def f_two_week_label(x, y, z):
            if len(str(x)) == 3:
                x = int(x[2])
            if len(str(y)) == 3:
                y = int(y[2])
            if max(x, y) < z:
                return -1
            elif max(x, y) >= z:
                return 1
        # df_two_week_all['label'] = df_two_week_all.apply(lambda x: f_two_week_label(x['module_id_x'], x['module_id_y'], x['max_module_load_x']), axis=1)
        ddf_two_week_all['label'] = ddf_two_week_all.apply(lambda x: f_two_week_label(x['module_id_x'], x['module_id_y'], x['max_module_load_x']), axis=1)

    if week_mode == 'one':
        if assign_mode == 'divided':
            # TODO: assign_divided_o
            mat = new4_middle1[[
                'duration_x', 'avg_prom_page_view_x', 'distinct_problem_view_count_x',
                'distinct_problem_attempt_count_x', 'distinct_problem_attempt_view_x',
                'submission_distinct_problem_attempt_x', 'submission_distinct_problem_correct_x', 'distinct_correct_count_x',
                'distinct_correct_attempt_x', 'grade_ration_x', 'duration_x',
                'avg_prom_page_view_y', 'distinct_problem_view_count_y',
                'distinct_problem_attempt_count_y', 'distinct_problem_attempt_view_y',
                'submission_distinct_problem_attempt_y', 'submission_distinct_problem_correct_y', 'distinct_correct_count_y',
                'distinct_correct_attempt_y', 'grade_ration_y', 'duration_y',
                'post_num', 'response_num', 'avg_post_length', 'avg_responded_num', 'duration_y',
                'active_days', 'page_view', 'event_std', 'browser_ratio', 'duration',
                'avg_view_number_video', 'avg_video', 'avg_view_number_session',
                'avg_only_load_video', 'avg_only_load_view', 'avg_straight_through',
                'avg_start_stop_num', 'avg_skip_ahead_num', 'avg_backward_num', 'avg_slow_rate_num',
                'avg_time_use', 'opened_finish_video_num', 'all_finish_video_num']]

        elif assign_mode == 'undivided':
            # TODO: assign_undivided_o
            mat = new4_middle1[[
                'duration_x', 'avg_prom_page_view', 'distinct_problem_view_count',
                'distinct_problem_attempt_count', 'distinct_problem_attempt_view',
                'submission_distinct_problem_attempt', 'submission_distinct_problem_correct', 'distinct_correct_count',
                'distinct_correct_attempt', 'grade_ration', 'duration_y', 'post_num', 'response_num',
                'avg_post_length', 'avg_responded_num', 'duration_x',
                'active_days', 'page_view', 'event_std', 'browser_ratio', 'duration_y',
                'avg_view_number_video', 'avg_video',
                'avg_view_number_session', 'avg_only_load_video', 'avg_only_load_view',
                'avg_straight_through', 'avg_start_stop_num', 'avg_skip_ahead_num',
                'avg_backward_num', 'avg_slow_rate_num', 'avg_time_use', 'opened_finish_video_num',
                'all_finish_video_num']]
        label = new4_middle1[['label']]

    if week_mode == 'two':
        if assign_mode == 'divided':
            mat = ddf_two_week_all[[
                 'duration_x_x', 'avg_prom_page_view_x_x',
                 'distinct_problem_view_count_x_x',
                 'distinct_problem_attempt_count_x_x',
                 'distinct_problem_attempt_view_x_x',
                 'submission_distinct_problem_attempt_x_x',
                 'submission_distinct_problem_correct_x_x',
                 'distinct_correct_count_x_x',
                 'distinct_correct_attempt_x_x', 'grade_ration_x_x', 'duration_x_x',
                 'avg_prom_page_view_y_x', 'distinct_problem_view_count_y_x',
                 'distinct_problem_attempt_count_y_x',
                 'distinct_problem_attempt_view_y_x',
                 'submission_distinct_problem_attempt_y_x',
                 'submission_distinct_problem_correct_y_x',
                 'distinct_correct_count_y_x', 'distinct_correct_attempt_y_x',
                 'grade_ration_y_x', 'duration_y_x', 'post_num_x', 'response_num_x',
                 'avg_post_length_x', 'duration_y_x',
                 'active_days_x', 'page_view_x', 'event_std_x', 'browser_ratio_x', 'duration_x',
                 'avg_view_number_video_x', 'avg_video_x', 'avg_view_number_session_x',
                 'avg_only_load_video_x', 'avg_only_load_view_x',
                 'avg_straight_through_x', 'avg_start_stop_num_x',
                 'avg_skip_ahead_num_x', 'avg_backward_num_x', 'avg_slow_rate_num_x',
                 'avg_time_use_x', 'opened_finish_video_num_x',
                 'all_finish_video_num_x', 'avg_responded_num_x', 'duration_x_y',
                 'avg_prom_page_view_x_y', 'distinct_problem_view_count_x_y',
                 'distinct_problem_attempt_count_x_y',
                 'distinct_problem_attempt_view_x_y',
                 'submission_distinct_problem_attempt_x_y',
                 'submission_distinct_problem_correct_x_y',
                 'distinct_correct_count_x_y', 'distinct_correct_attempt_x_y',
                 'grade_ration_x_y', 'duration_x_y', 'avg_prom_page_view_y_y',
                 'distinct_problem_view_count_y_y',
                 'distinct_problem_attempt_count_y_y',
                 'distinct_problem_attempt_view_y_y',
                 'submission_distinct_problem_attempt_y_y',
                 'submission_distinct_problem_correct_y_y',
                 'distinct_correct_count_y_y', 'distinct_correct_attempt_y_y',
                 'grade_ration_y_y', 'duration_y_y', 'post_num_y', 'response_num_y',
                 'avg_post_length_y', 'duration_y_y',
                 'active_days_y', 'page_view_y', 'event_std_y', 'browser_ratio_y', 'duration_y',
                 'avg_view_number_video_y', 'avg_video_y', 'avg_view_number_session_y',
                 'avg_only_load_video_y', 'avg_only_load_view_y',
                 'avg_straight_through_y', 'avg_start_stop_num_y',
                 'avg_skip_ahead_num_y', 'avg_backward_num_y', 'avg_slow_rate_num_y',
                 'avg_time_use_y', 'opened_finish_video_num_y',
                 'all_finish_video_num_y', 'avg_responded_num_y']]

        elif assign_mode == 'undivided':
            mat = ddf_two_week_all[[
                'duration_x_x', 'avg_prom_page_view_x', 'distinct_problem_view_count_x',
                'distinct_problem_attempt_count_x', 'distinct_problem_attempt_view_x',
                'submission_distinct_problem_attempt_x', 'submission_distinct_problem_correct_x', 'distinct_correct_count_x',
                'distinct_correct_attempt_x', 'grade_ration_x', 'duration_y_x', 'post_num_x',
                'response_num_x', 'avg_post_length_x', 'duration_x_x',
                'active_days_x', 'page_view_x', 'event_std_x', 'browser_ratio_x', 'duration_y_x',
                'avg_view_number_video_x', 'avg_video_x', 'avg_view_number_session_x',
                'avg_only_load_video_x', 'avg_only_load_view_x', 'avg_straight_through_x',
                'avg_start_stop_num_x', 'avg_skip_ahead_num_x', 'avg_backward_num_x',
                'avg_slow_rate_num_x', 'avg_time_use_x', 'opened_finish_video_num_x',
                'all_finish_video_num_x', 'avg_responded_num_x', 'duration_x_y',
                'avg_prom_page_view_y', 'distinct_problem_view_count_y',
                'distinct_problem_attempt_count_y', 'distinct_problem_attempt_view_y',
                'submission_distinct_problem_attempt_y', 'submission_distinct_problem_correct_y', 'distinct_correct_count_y',
                'distinct_correct_attempt_y', 'grade_ration_y', 'duration_y_y', 'post_num_y',
                'response_num_y', 'avg_post_length_y', 'duration_x_y',
                'active_days_y', 'page_view_y', 'event_std_y', 'browser_ratio_y', 'duration_y_y',
                'avg_view_number_video_y', 'avg_video_y', 'avg_view_number_session_y',
                'avg_only_load_video_y', 'avg_only_load_view_y', 'avg_straight_through_y',
                'avg_start_stop_num_y', 'avg_skip_ahead_num_y', 'avg_backward_num_y',
                'avg_slow_rate_num_y', 'avg_time_use_y', 'opened_finish_video_num_y',
                'all_finish_video_num_y', 'avg_responded_num_y']]

        # label = df_two_week_all[['label']]
        label = ddf_two_week_all[['label']]

    # print(mat.shape)
    return mat, label

# if __name__ == '__main__':
#     courses = [
#         '102_1x_4T2015',
#         '101x_1T2016',
#         '102x_1T2016'
#     ]

    # classifiers = [
    #     # SVC(kernel="linear", C=1),
    #     SVC(kernel="linear", C=0.025),
    #     # SVC(gamma=2, C=0.025),
    #     LogisticRegression(C=1.0),
    #     # DecisionTreeClassifier(max_depth=5),
    #     # GaussianNB(),
    #     # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    #     KNeighborsClassifier(3),
    #     AdaBoostClassifier(),
    #     MLPClassifier(alpha=1)
    #     # GaussianProcessClassifier(1.0 * RBF(1.0))
    #     # QuadraticDiscriminantAnalysis()
    # ]
    #
    # user_modes = [
    #     'active',
    #     'passive',
    #     'all'
    # ]
    #
    # for classifier in classifiers:
    #     for course in courses:
    #         for user_mode in user_modes:
    #             mat, label = file2mat(course_id = course, assign_mode = 'undivided', user_mode = user_mode, user_mode_threshold = 2, week_mode = 'one', video_mode = 'threshold')
    #             list_mat = mat.fillna(0).values.tolist()
    #             emnlp_mat = np.asarray(list_mat, dtype=np.float64)
    #             list_label = label['label'].tolist()
    #             emnlp_label = np.array(list_label)
    #
    #             emnlp_mat[np.isinf(emnlp_mat) == True] = 0
    #             min_max_scaler = preprocessing.MinMaxScaler(copy=True, feature_range=(0, 1))
    #             X_minmax = min_max_scaler.fit_transform(emnlp_mat)
    #
    #             accuracy = cross_val_score(classifier, X=X_minmax,y=emnlp_label,cv=5,scoring=make_scorer(accuracy_score))
    #             precision = cross_val_score(classifier, X=X_minmax,y=emnlp_label,cv=5,scoring=make_scorer(precision_score))
    #             recall = cross_val_score(classifier, X=X_minmax,y=emnlp_label,cv=5,scoring=make_scorer(recall_score))
    #             f1 = cross_val_score(classifier, X=X_minmax,y=emnlp_label,cv=5,scoring=make_scorer(f1_score))
    #             print ('Accuracy', reduce(lambda x, y: x + y, accuracy) / len(accuracy))
    #             print ('Precision', reduce(lambda x, y: x + y, precision) / len(precision))
    #             print ('Recall', reduce(lambda x, y: x + y, recall) / len(recall))
    #             print ('F1 score', reduce(lambda x, y: x + y, f1) / len(f1))
    #
    #             with open('/Users/xuyujie/Desktop/result_single_4T2015.txt', 'a') as out:
    #                 out.write("%s %s %s %s %s \n" % (course, 'undivided', user_mode, 'one', classifier))
    #                 out.write("%s %s\n" % ('Accuracy', str(reduce(lambda x, y: x + y, accuracy) / len(accuracy))))
    #                 out.write("%s %s\n" % ('Precision', str(reduce(lambda x, y: x + y, precision) / len(precision))))
    #                 out.write("%s %s\n" % ('Recall', str(reduce(lambda x, y: x + y, recall) / len(recall))))
    #                 out.write("%s %s\n" % ('F1 score', str(reduce(lambda x, y: x + y, f1) / len(f1))))
    #                 out.write("%s\n" % (str(X_minmax.shape)))
    # out.close()

    # for course in courses:
    #     file2mat(course_id = course, assign_mode = 'undivided', user_mode = 'active', user_mode_threshold = 2, week_mode = 'two', video_mode = 'threshold')


#     mat, label = file2mat(course_id = '101x_1T2016', assign_mode = 'undivided', user_mode = 'active', user_mode_threshold = 2, week_mode = 'two', video_mode = 'threshold')
#     list_mat = mat.fillna(0).values.tolist()
#     emnlp_mat = np.asarray(list_mat, dtype=np.float64)
#     list_label = label['label'].tolist()
#     emnlp_label = np.array(list_label)
#
#     emnlp_mat[np.isinf(emnlp_mat) == True] = 0
#     min_max_scaler = preprocessing.MinMaxScaler(copy=True, feature_range=(0, 1))
#     X_minmax = min_max_scaler.fit_transform(emnlp_mat)
#
#     clf = svm.SVC(kernel='linear', C=1)
#     accuracy = cross_val_score(clf, X=X_minmax,y=emnlp_label,cv=5,scoring=make_scorer(accuracy_score))
#     precision = cross_val_score(clf, X=X_minmax,y=emnlp_label,cv=5,scoring=make_scorer(precision_score))
#     recall = cross_val_score(clf, X=X_minmax,y=emnlp_label,cv=5,scoring=make_scorer(recall_score))
#     f1 = cross_val_score(clf, X=X_minmax,y=emnlp_label,cv=5,scoring=make_scorer(f1_score))
#     print ('Accuracy', reduce(lambda x, y: x + y, accuracy) / len(accuracy))
#     print ('Precision', reduce(lambda x, y: x + y, precision) / len(precision))
#     print ('Recall', reduce(lambda x, y: x + y, recall) / len(recall))
#     print ('F1 score', reduce(lambda x, y: x + y, f1) / len(f1))

    # TODO: draw plot to select features
    # from sklearn.ensemble import RandomForestClassifier
    # rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
    # rnd_clf.fit(X_minmax_1021x4T2015, emnlp_label_1021x4T2015)
    # importances = rnd_clf.feature_importances_
    # std = np.std([tree.feature_importances_ for tree in rnd_clf.estimators_], axis=0)
    # indices = np.argsort(importances)[::-1]
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.title("Feature Importances")
    # plt.bar(range(X_minmax_1021x4T2015.shape[1]), importances[indices],
    #        color="r", yerr=std[indices], align="center")
    # plt.xticks(range(X_minmax_1021x4T2015.shape[1]), indices)
    # plt.xlim([-1, X_minmax_1021x4T2015.shape[1]])
    # plt.show()
