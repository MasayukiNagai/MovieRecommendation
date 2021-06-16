from surprise import Reader
from surprise import Dataset
from surprise import SVD, SVDpp
from surprise.model_selection import cross_validate
import csv
import pandas as pd
import numpy as np
from collections import defaultdict

with open('CompetitionDataFinal/movie-codes.txt') as f:
    movie_dict = {}
    for line in f.readlines():
        line.rstrip()
        value, key = line.split('\t')
        key = int(key.rstrip())
        movie_dict[key] = value

impression_df = pd.read_csv('CompetitionDataFinal/impressions-train.csv')
with open('CompetitionDataFinal/impressions-train.csv') as f:
    reader = csv.reader(f)
    impression_data = list(reader)
    impression_dict = defaultdict(list)
    for reviewerid, movie_code, rating in impression_data[1:]:
        impression_dict[int(reviewerid)].append((int(movie_code), int(rating)))
#    impression_df = pd.DataFrame(data = list(map(int, impression_data[1:])), columns =impression_data[0])

ratings_df = pd.read_csv('CompetitionDataFinal/ratings-final.csv')
with open('CompetitionDataFinal/ratings-final.csv') as f:
    reader = csv.reader(f)
    ratings_data = list(reader)
    ratings_dict = defaultdict(list)
    for reviewerid, movie_code, rating in ratings_data[1:]:
        ratings_dict[int(reviewerid)].append((int(movie_code), int(rating)))
ratings_df = pd.DataFrame(data=ratings_data[1:], columns=ratings_data[0])

test_df = pd.read_csv('CompetitionDataFinal/test.csv')
with open('CompetitionDataFinal/test.csv') as f:
    reader = csv.reader(f)
    test_data = list(reader)
    test_df = pd.DataFrame(data=test_data[1:], columns=test_data[0])

reader = Reader(rating_scale = (0,2))
impression_ds = Dataset.load_from_df(impression_df, reader = reader)
ratings_ds = Dataset.load_from_df(ratings_df, reader = reader)

# a = impression_df.loc[(impression_df['reviewerid'] == 0) & (impression_df['movie-code'] == 7)]['rating'].tolist()
# print(a)

# algorithm_ratings = SVD()
algorithm_ratings = SVDpp()
algorithm_ratings.fit(ratings_ds.build_full_trainset())
# cross_validate(algorithm,ratings_ds, measures=['RMSE', 'MAE'], cv=5, verbose=True)
testset_ratings = ratings_ds.build_full_trainset().build_anti_testset()
pre_ratings = algorithm_ratings.test(testset_ratings)

# algorithm_imp = SVD()
algorithm_imp = SVDpp()
algorithm_imp.fit(impression_ds.build_full_trainset())
testset_imp = impression_ds.build_full_trainset().build_anti_testset()
pre_imp = algorithm_ratings.test(testset_imp)

def get_expected_rating(predictions):
    exp_dict = defaultdict(list)
    for rid, mid, true_r, est, _ in predictions:
        exp_dict[int(rid)].append((int(mid), est))
    for rid, user_ratings in exp_dict.items():
        user_ratings.sort(key=lambda x: x[0], reverse=False)
        exp_dict[rid] = user_ratings    
    return exp_dict

def insertRID(ls, rid):
    ls.insert(0, rid)
    return ls

def get_list_expected_rating(predictions):
    exp_dict = defaultdict(list)
    exp_list = []
    for rid, mid, true_r, est, _ in predictions:
        exp_dict[int(rid)].append([int(mid), float(est)])
    for rid, user_ratings in exp_dict.items():
        user_ratings.sort(key=lambda x: x[0], reverse=False)
        exp_list += (list(map(lambda x:insertRID(x, rid), user_ratings)))
    return exp_list

# exp_ratings = get_expected_rating(pre_ratings)
# exp_imp = get_expected_rating(pre_imp)

exp_ratings_list = get_list_expected_rating(pre_ratings)
exp_imp_list = get_list_expected_rating(pre_imp)

df_exp_rating = pd.DataFrame(data=exp_ratings_list, columns=['reviewerid', 'movie-code', 'rating'])
df_exp_imp = pd.DataFrame(data=exp_imp_list, columns=['reviewerid', 'movie-code', 'rating'])
# print(df_exp_rating)
# print(df_exp_imp)

def diff():
    print('start diff function')
    column_names = ['reviewerid', 'movie-code', 'rating']
    lsofls = []
    rnum = 572
    mnum = 200
    for i in range(0, rnum+1):
        rated_movie = ratings_df.loc[ratings_df.reviewerid == i]['movie-code'].tolist()
        imp_movie = impression_df.loc[impression_df.reviewerid == i]['movie-code'].tolist()
        for j in range(0, mnum+1):
            if j in rated_movie:
                try:
                    rate_act = ratings_df.loc[(ratings_df.reviewerid == i) & (ratings_df['movie-code'] == j)]['rating'].tolist()[0]
                    imp_est = df_exp_imp.loc[(df_exp_imp.reviewerid == i) & (df_exp_imp['movie-code'] == j)]['rating'].tolist()[0]
                    difference = rate_act - imp_est
                    diff_ls = [i, j, difference]
                    lsofls.append(diff_ls)
                except:
                    print('cannot find ', i, j)
            elif j in imp_movie:
                try:
                    imp_act = impression_df.loc[(impression_df.reviewerid == i) & (impression_df['movie-code'] == j)]['rating'].tolist()[0]
                    rate_est = df_exp_rating.loc[(df_exp_rating.reviewerid == i) & (df_exp_rating['movie-code'] == j)]['rating'].tolist()[0]
                    difference = rate_est - imp_act
                    diff_ls = [i, j, difference]
                    lsofls.append(diff_ls)
                except:
                    print('cannot find ', i, j)
        print(i)
    df = pd.DataFrame(lsofls, columns = column_names)  
    return df

# df_diff = diff()
# df_diff.to_csv('CompetitionDataFinal/difference.csv', index = False, header = True)
# print(df_diff)

df_diff = pd.read_csv('CompetitionDataFinal/difference.csv')
print(df_diff)

diff_ds = Dataset.load_from_df(df_diff, reader = reader)
# algorithm_diff = SVD()
algorithm_diff = SVDpp()
algorithm_ratings.fit(diff_ds.build_full_trainset())
cross_validate(algorithm_diff,diff_ds, measures=['RMSE', 'MAE'], cv=5, verbose=True)
testset_diff = diff_ds.build_full_trainset().build_anti_testset()
pre_diff = algorithm_diff.test(testset_diff)

est_diff_list = get_list_expected_rating(pre_diff)
df_est_diff = pd.DataFrame(est_diff_list, columns=['reviewerid', 'movie-code', 'rating'])
print(df_est_diff)

def calcImpression(a, b):
    print('start estimating impression')
    column_names = ['reviewerid', 'movie-code', 'rating']
    lsofls = []
    for index, row in test_df.iterrows():
        rid = int(row['reviewerid'])
        mid = int(row['movie-code'])
        try: 
            rate_est = df_exp_rating.loc[(df_exp_rating.reviewerid == rid) & (df_exp_rating['movie-code'] == mid)]['rating'].tolist()[0]
            diff_est = df_est_diff.loc[(df_est_diff.reviewerid == rid) & (df_est_diff['movie-code'] == mid)]['rating'].tolist()[0]
            if diff_est < 0:
                diff = rate_est - b * diff_est
            else:
                diff = rate_est - a * diff_est
            value = round(diff)
            if value == 0 or value == 1 or value == 2:
                imp_est = value
            elif value < 0:
                imp_est = 0
            else:
                imp_est = 2
        except:
            print('cannot find: ', rid, mid)
            diff = 999
            imp_est = 0
        lsofls.append([rid, mid, imp_est])
    df = pd.DataFrame(lsofls, columns = column_names)
    return df

a = 1.2
b = 1.2
result_df = calcImpression(a, b)
result_df.to_csv('CompetitionDataFinal/test_est.csv', index = False, header = True)