from facility_location import FacilityLocation
from greedy import greedy
from instance_upb import dual

import random

model = FacilityLocation(matrix_path="/home/ctong/Projects/SubOptKnapsack/dataset/movie/movie_by_user_small_rating_rank_norm.npy", k=2)
res = greedy(model)
print(res)


N = model.ground_set
k = 10  # size of guess collection
b = 3  # initial set size
guess_collection = [ random.sample(N, 3) for _ in range(b) ]

upb = dual(model, guess_collection)


print(upb)