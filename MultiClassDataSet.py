import math

from utils import make_examples

class MultiClassDataSet:
    def __init__(self, attributes, classes, examples):
        # attributes = ['A0', 'A1', 'A2', 'A3')]
        # classes = 3 => {0, 1, 2}
        # examples = [((1.1125, ...), 1), ((0.0852, ...), 2), ...]
        self.attributes = attributes
        self.classes = classes
        self.examples = examples
        self.attribute_indexes = self.index_attributes(attributes)
        self.distribution = self.calculate_class_distribution()
        self.H = self.calculate_entropy(self.distribution)
        self.GI = self.calculate_gini_impurity(self.distribution)

    def index_attributes(self, attributes):
        attribute_indexes = dict()
        for i in  range(len(attributes)):
            V = attributes[i]
            attribute_indexes[V] = i
        return attribute_indexes
    
    def calculate_class_distribution(self):
        counts = [0 for i in range(self.classes)]
        for ex in self.examples:
            (attrs, y) = ex
            counts[y] += 1
        distribution = self.normalize(counts)
        return distribution

    def normalize(self, counts):
        denominator = sum(counts)
        normalized = [x / denominator for x in counts]
        return normalized

    def calculate_entropy(self, distribution):
        H = 0
        for j in range(len(distribution)):
            p_c_j = distribution[j]
            if p_c_j > 0:
                H += p_c_j * math.log2(1 / p_c_j)
        return H

    def calculate_gini_impurity(self, distribution):
        GI = 1
        for j in range(len(distribution)):
            p_j = distribution[j]
            GI -= (p_j * p_j)
        return GI

    def partition(self, A, split_point):
        A_index = self.attribute_indexes[A]
        partitioned_examples = [[], []]
        for ex in self.examples:
            (attrs, y) = ex
            v = attrs[A_index]
            if v < split_point:
                partitioned_examples[0].append(ex)
            else:
                partitioned_examples[1].append(ex)
        return partitioned_examples

    def get_split_points(self, A):
        A_index = self.attribute_indexes[A]
        attr_values = list(map(lambda ex: ex[0][A_index], self.examples))
        attr_values.sort()
        split_points = []
        for i in range(len(attr_values) - 1):
            split_point = ( attr_values[i] + attr_values[i + 1] ) / 2
            split_points.append(split_point)
        return split_points


    def calculate_remaining_entropy(self, A, split_point):
        partitioned_examples = self.partition(A, split_point)
        left_data_set = MultiClassDataSet(self.attributes, self.classes, partitioned_examples[0])
        right_data_set = MultiClassDataSet(self.attributes, self.classes, partitioned_examples[1])
        left_weight = len(left_data_set.examples) / len(self.examples)
        right_weight = len(right_data_set.examples) / len(self.examples)
        # print("Inside calculate_remaining_entropy, the left_data_set has entropy %s, and examples: " % (left_data_set.H))
        # print(left_data_set.examples)

        # print("Inside calculate_remaining_entropy, the right_data_set has entropy %s, and examples: " % (right_data_set.H))
        # print(right_data_set.examples)
        remainder = left_weight * left_data_set.H + right_weight * right_data_set.H
        return remainder

    def calculate_remaining_gini_impurity(self, A, split_point):
        partitioned_examples = self.partition(A, split_point)
        left_data_set = MultiClassDataSet(self.attributes, self.classes, partitioned_examples[0])
        right_data_set = MultiClassDataSet(self.attributes, self.classes, partitioned_examples[1])
        print("Inside calculate_remaining_gini_impurity, got left_data_set and right_data_set:")
        print(left_data_set.examples)
        print(right_data_set.examples)
        left_weight = len(left_data_set.examples) / len(self.examples)
        right_weight = len(right_data_set.examples) / len(self.examples)
        remainder = left_weight * left_data_set.GI + right_weight * right_data_set.GI
        return remainder

    def calculate_gain(self, A, split_point):
        remainder = self.calculate_remaining_entropy(A, split_point)
        gain = self.H - remainder
        return gain

    def calculate_gini_gain(self, A, split_point):
        remainder = self.calculate_remaining_gini_impurity(A, split_point)
        gini_gain = self.GI - remainder
        return gini_gain

    def find_best_attribute_split_point(self, use_gini=False):
        best_gain = float("-inf")
        best_attr = None
        best_attr_split = None
        for A in self.attributes:
            A_split_points = self.get_split_points(A)
            for A_split_point in A_split_points:
                if use_gini:
                    this_gain = self.calculate_gini_gain(A, A_split_point)
                else:
                    this_gain = self.calculate_gain(A, A_split_point)
                if this_gain > best_gain:
                    best_gain = this_gain
                    best_attr = A
                    best_attr_split = A_split_point
        return (best_attr, best_attr_split)
