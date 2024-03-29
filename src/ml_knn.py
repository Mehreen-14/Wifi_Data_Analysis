import json
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt

import numpy

from heuristics import euclidean_distance
from heuristics import Manhattan_distance
from heuristics import Chebyshev_distance
from heuristics import Simple_matching_coefficient
from heuristics import DotProduct
from heuristics import Cosine_similarity
from heuristics import Jaccard_similarity



from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Open the JSON file and load the data
with open('../data/203_ALL.json', 'r') as file:
    data = json.load(file)



#room_pos_no is inside or outside    

testPointsInside = []
testPointsOutside = []
refPointsInside = []
refPointsOutside = []

for room_pos, inner_data in data.items():
    if 'testing' in room_pos:
        if room_pos.endswith('_i'):
            testPointsInside.append(room_pos)
        else:
            testPointsOutside.append(room_pos)
    else:
        if room_pos.endswith('_i'):
            refPointsInside.append(room_pos)
        else:
            refPointsOutside.append(room_pos)

# Print the lists for verification
# print("Test Points Inside:", testPointsInside)
# print("Test Points Outside:", testPointsOutside)
# print("Reference Points Inside:", refPointsInside)
# print("Reference Points Outside:", refPointsOutside)

# Function to calculate average strength of a given SSID
selectedSSID = [
    "Galaxy M124213",
    "CSE-204",
    "CSE-303",
    "CSE-304",
    "Hall of Fame",
    "DataLab@BUET",
    "CSE-206",
    "CSE-G04",
    "CSE-G07",
    "CSE-306",
    "dlink",
    "CSE-401"
]



# Function to remove duplicate SSIDs
def remove_duplicates(ssids):
    return list(set(ssids))

# Remove duplicates from selectedSSID
selectedSSID = remove_duplicates(selectedSSID)


# Function to remove outliers using the IQR method
def remove_outliers(values):
    # Calculate the first and third quartiles
    Q1 = numpy.percentile(values, 25)
    Q3 = numpy.percentile(values, 75)
    
    # Calculate the IQR
    IQR = Q3 - Q1
    
    # Define the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter out values outside the bounds
    filtered_values = [val for val in values if val >= lower_bound and val <= upper_bound]
    
    return filtered_values

def avg_strength_for_ssid(data, ssid):
    strengths = []

    for scan_result in data.values():
        for wifi_list in scan_result["ScanList"]:
            for wifi_info in wifi_list:
                if wifi_info["SSID"] == ssid:
                    strengths.append(wifi_info["Strength"])
    
    # Check if strengths array is empty
    if not strengths:
        total_strength = -80  # Assuming -80 for no value found
        count = 1  # Setting count to 1 to avoid division by zero
    else:
        # Remove outliers using the IQR method
        filtered_strengths = remove_outliers(strengths)
    
        # Calculate average strength
        total_strength = sum(filtered_strengths)
        count = len(filtered_strengths)
    
    return total_strength / count, count


# Define the reference and testing points
refPoints = refPointsInside + refPointsOutside
testingPoints = testPointsInside + testPointsOutside


# Organize reference and test points into dictionaries
ref_points = {room_pos: {ssid: avg_strength_for_ssid(data[room_pos], ssid)[0] for ssid in selectedSSID} for room_pos in refPoints}
test_points = {room_pos: {ssid: avg_strength_for_ssid(data[room_pos], ssid)[0] for ssid in selectedSSID} for room_pos in testingPoints}

def knn_with_majority_voting(ref_points, test_points, k):
    X_train = []  # Features
    y_train = []  # Target labels

    for test_key, test_value in test_points.items():
        distance_map = {}
        for ref_key, ref_value in ref_points.items():
            distance_map[ref_key] = DotProduct(ref_value, test_value)
        sorted_distances = sorted(distance_map.items(), key=lambda x: x[1])[:k]  # Select k nearest neighbors
        distances = [dist for _, dist in sorted_distances]  # Extract distances
        X_train.append(distances)
        y_train.append(1 if test_key.endswith("_i") else 0)  # 1 for inside, 0 for outside

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Initialize and train the logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy:", accuracy)

# Define the value of k
k = 7 # Choose an appropriate value for k

# Run the KNN algorithm
knn_with_majority_voting(ref_points, test_points, k)
