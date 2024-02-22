import json
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt

import numpy

# Open the JSON file and load the data
with open('../data/204_ALL.json', 'r') as file:
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

def avg_strength_for_ssid(data, ssid):
    total_strength = 0
    count = 0
    for scan_result in data.values():
        for wifi_list in scan_result["ScanList"]:
            for wifi_info in wifi_list:
                if wifi_info["SSID"] == ssid:
                    total_strength += wifi_info["Strength"]
                    count += 1
    if count == 0:
        return 0, 0
    return total_strength / count, count

# Iterate through each room position
# for room_position, room_data in data.items():
#     print("Room Position:", room_position)
#     # Calculate average strength and count for each SSID in selectedSSID
#     for ssid in selectedSSID:
#         avg_strength, count = avg_strength_for_ssid(room_data, ssid)
#         print(ssid, ":", '{:.4f}'.format(avg_strength), "Count:", count)


#Euclidean distance
def euclidean_distance(ref_point, test_point):
    distance = 0.0
    for ssid, ref_strength in ref_point.items():
        test_strength = test_point.get(ssid, 0.0)  # Use 0.0 if the SSID is not found in the test point
        distance += (ref_strength - test_strength) ** 2
    return distance ** 0.5

refPoints = refPointsInside + refPointsOutside
testingPoints = testPointsInside + testPointsOutside


# Organize reference and test points into dictionaries
ref_points = {room_pos: {ssid: avg_strength_for_ssid(data[room_pos], ssid)[0] for ssid in selectedSSID} for room_pos in refPoints}
test_points = {room_pos: {ssid: avg_strength_for_ssid(data[room_pos], ssid)[0] for ssid in selectedSSID} for room_pos in testingPoints}

def knn_with_majority_voting(ref_points, test_points, k):
    matched_count = 0
    unmatched_count = 0
    for test_key, test_value in test_points.items():
        distance_map = {}
        for ref_key, ref_value in ref_points.items():
            distance_map[ref_key] = euclidean_distance(ref_value, test_value)
        sorted_distances = sorted(distance_map.items(), key=lambda x: x[1])
        count_map = {"inside": 0, "outside": 0}
        for i in range(k):
            ref_point, _ = sorted_distances[i]
            if ref_point.endswith("_i"):
                print("Inside:", ref_point, "Distance:", distance_map[ref_point])
                count_map["inside"] += 1
            else:
                print("Outside:", ref_point, "Distance:", distance_map[ref_point])
                count_map["outside"] += 1
        if count_map["inside"] > count_map["outside"]:
            if test_key.endswith("_i"):
                print(test_key, "is inside (Correct)")
                matched_count += 1
            else:
                print(test_key, "is inside (Incorrect)")
                unmatched_count += 1
        else:
            if test_key.endswith("_i"):
                print(test_key, "is outside (Incorrect)")
                unmatched_count += 1
            else:
                print(test_key, "is outside (Correct)")
                matched_count += 1
    print("Matched:", matched_count)
    print("Unmatched:", unmatched_count)

# Define the value of k
k = 7 # Choose an appropriate value for k

# Run the KNN algorithm
knn_with_majority_voting(ref_points, test_points, k)
