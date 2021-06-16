from surprise import Reader
from surprise import Dataset
from surprise import SVD

# with open('ml-100k/u.data') as f:
#     for _ in range(10):
#         print(f.readline(), end = '')

reader = Reader(line_format='user item rating timestamp', sep = '\t')
data = Dataset.load_from_file('ml-100k/u.data', reader = reader)
print('Frist ten users and their ratings for a movie')
data.build_full_trainset().ir[0][:10]

algorithm = SVD()
algorithm.fit(data.build_full_trainset())
print(algorithm.predict(0, 1))