from application.movie_recommendation_knapsack import MovieRecommendationKnapsack
from application.image_sum_knapsack import ImageSummarizationKnapsack
from application.facility_location_knapsack import MaximumFacilityLocationKnapsack
from application.revenue_max_knapsack import RevenueMaxKnapsack

def model_constructor(model_name: str):
    if model_name == "MovieRecommendation":
        model = MovieRecommendationKnapsack(n=50, k=40, budget_ratio=0.2)
    elif model_name == "ImageSum":
        model = ImageSummarizationKnapsack(
            budget_ratio=0.3, image_path="application/dataset/image/500_cifar10_sample.npy", max_num=50)
    elif model_name == "MaxFLP":
        model = MaximumFacilityLocationKnapsack(num_facilities=50, num_customers=40)
    elif model_name == "MaxRevenue":
        model = RevenueMaxKnapsack(budget=5, pckl_path="application/dataset/revenue/25_youtube_top5000.pkl")
    else:
        raise ValueError("Model name is not valid")
    return model