import numpy as np

def entropy(labels):
    """Compute entropy of the data with labels."""
    # YOUR CODE HERE
    unique_labels, freq = np.unique(labels, return_counts=True)
    prob = freq / len(labels) # prob = probability array of elements
    entropy = -np.sum(prob * np.log2(prob))
    return entropy

def compute_information_gain(data, feature_index, labels):
    """
    Computes information gain of a feature for a given dataset.

    Parameters:
    data (np.array): Feature matrix of shape (n_samples, n_features)
    feature_index (int): Index of the feature to compute IG
    labels (list or np.array): List of class labels

    Returns:
    float: Information gain
    """
    sum=np.sum(data,axis=0)
    thresh=sum[feature_index]/data.shape[0] #mean of given features #answer for Q8
    split1=[]
    split2=[]
    for i, point in enumerate(data):
        if point[feature_index]<=thresh:
            split1.append(labels[i])
        else:
            split2.append(labels[i])
    split1=np.array(split1)
    split2=np.array(split2)
    info_gain=entropy(labels)-(len(split1)/len(labels))*entropy(split1)-(len(split2)/len(labels))*entropy(split2)
    return info_gain,thresh

def best_split(data, labels):
    """Find the best feature and threshold for splitting."""
    best_info_gain = -1
    best_feature = None
    best_threshold = None
    for feature_index in range(data.shape[1]):
        info_gain, threshold = compute_information_gain(data, feature_index, labels)
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = feature_index
            best_threshold = threshold
    return best_feature, best_threshold
    # YOUR CODE HERE
    raise NotImplementedError()



class DecisionTreeNode:
    """Class for a decision tree node."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.root = None

    def fit(self, data, labels, depth=0):
        """Recursively builds the decision tree."""
        # Base case: Stop splitting if pure or max depth reached
        if depth == self.max_depth or len(np.unique(labels)) == 1: #Why?
            return DecisionTreeNode(value=np.bincount(labels).argmax()) #Why?

        # Find best feature and threshold
        best_feature, best_threshold = best_split(data, labels)

        # Partition data
        left_mask = data[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        if np.all(left_mask) or np.all(right_mask):  # Prevent infinite splitting
            return DecisionTreeNode(value=np.bincount(labels).argmax())

        left_child = self.fit(data[left_mask], labels[left_mask], depth + 1)
        right_child = self.fit(data[right_mask], labels[right_mask], depth + 1)

        return DecisionTreeNode(feature=best_feature, threshold=best_threshold, left=left_child, right=right_child)

    def train(self, data, labels):
        """Initialize tree training."""
        self.root = self.fit(data, labels, depth=0)

def predict_sample(node, sample):
    """Recursively predict the label for a single sample."""
    if node.value is not None:
        return node.value
    if sample[node.feature] <= node.threshold:
        return predict_sample(node.left, sample)
    else:
        return predict_sample(node.right, sample)

def predict(tree, data):
    """Predict labels for multiple samples."""
    return np.array([predict_sample(tree.root, sample) for sample in data])