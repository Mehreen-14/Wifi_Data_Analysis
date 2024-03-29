#Euclidean distance
def euclidean_distance(ref_point, test_point):
    distance = 0.0
    for ssid, ref_strength in ref_point.items():
        test_strength = test_point.get(ssid, 0.0)  # Use 0.0 if the SSID is not found in the test point
        distance += (ref_strength - test_strength) ** 2
    return distance ** 0.5

def Manhattan_distance(ref_point, test_point):
    distance = 0.0
    for ssid, ref_strength in ref_point.items():
        test_strength = test_point.get(ssid, 0.0)  # Use 0.0 if the SSID is not found in the test point
        distance += abs(ref_strength - test_strength)
    return distance

def Chebyshev_distance(ref_point, test_point):
    distance = 0.0
    for ssid, ref_strength in ref_point.items():
        test_strength = test_point.get(ssid, 0.0)  # Use 0.0 if the SSID is not found in the test point
        distance = max(distance, abs(ref_strength - test_strength))
    return distance


def Simple_matching_coefficient(ref_point, test_point):
    distance = 0.0
    for ssid, ref_strength in ref_point.items():
        test_strength = test_point.get(ssid, 0.0)  # Use 0.0 if the SSID is not found in the test point
        if ref_strength != test_strength:
            distance += 1
    return distance


def DotProduct(ref_point, test_point):
    distance = 0.0
    for ssid, ref_strength in ref_point.items():
        test_strength = test_point.get(ssid, 0.0)  # Use 0.0 if the SSID is not found in the test point
        distance += ref_strength * test_strength
    return distance


def Cosine_similarity(ref_point, test_point):
    distance = 0.0
    ref_length = 0.0
    test_length = 0.0
    for ssid, ref_strength in ref_point.items():
        test_strength = test_point.get(ssid, 0.0)  # Use 0.0 if the SSID is not found in the test point
        distance += ref_strength * test_strength
        ref_length += ref_strength ** 2
        test_length += test_strength ** 2
    return distance / ((ref_length ** 0.5) * (test_length ** 0.5))

def Jaccard_similarity(ref_point, test_point):
    distance = 0.0
    ref_length = 0.0
    test_length = 0.0
    for ssid, ref_strength in ref_point.items():
        test_strength = test_point.get(ssid, 0.0)  # Use 0.0 if the SSID is not found in the test point
        distance += min(ref_strength, test_strength)
        ref_length += ref_strength
        test_length += test_strength
    return distance / (ref_length + test_length - distance)
