from movie_recommendation import MovieRecommendation


def test():
    model = MovieRecommendation(
        matrix_path="./dataset/movie/user_by_movies_small_rating.npy", budget=181, k=30, n=500, knapsack=True,
        prepare_max_pair=False, print_curvature=False)


    pass

