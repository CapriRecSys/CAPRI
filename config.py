import os

# Paths
dataDirectory = os.path.abspath('./Data/')
outputsDir = os.path.abspath('./Outputs/')

# Default Parameters
topK = 10  # Top-k items to evaluate (default: 10)
limitUsers = -1  # Limit the number of users (default: -1)
listLimit = 15  # Limit the length of recommendation list (default: 10)
activeUsersPercentage = [5, 20]  # Calculate [n] percents of users as active

# Key: Model name, Value: Covered Contexts
models = {
    "GeoSoCa": ["Geographical", "Social", "Categorical"],
    "LORE": ["Geographical", "Social", "Temporal"],
    "USG": ["Interaction", "Social", "Geographical"],
}

# Key: Dataset name, Value: Covered Contexts
datasets = {
    "Gowalla": ["Geographical", "Social", "Temporal", "Interaction"],
    "Yelp":  ["Geographical", "Social", "Temporal", "Categorical", "Interaction"],
}

# An array of selected operations
# TODO: "WeightedSum" is not implemented yet
fusions = ["Product", "Sum"]

# List of evaluation metrics
evaluationMetrics = ["Precision", "Recall", "mAP", "NDCG"]

# Models Dictionaries
GeoSoCaDict = {"alpha": 0.5}
LoreDict = {"alpha": 0.05, "deltaT": 3600 * 24}
USGDict = {"alpha": 0.1, "beta": 0.1, "eta": 0.05}
