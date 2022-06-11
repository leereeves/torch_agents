import numpy as np 

class SumTree():
    def __init__(self, capacity):
        self.capacity = capacity
        # The leaves of this tree hold the weights of data elements in
        # locations (capacity - 1) to (2 * capacity - 2).
        # the branches of this tree hold a binary tree of cumulative 
        # sums of weights in elements 0 to (capacity - 2).
        # The parent of a node at index i is in location (i-1)//2.
        #                       0
        #           1                         2
        #      3          4             5           6
        #    7   8      9   10       11   12     13   14
        # For example, if capacity were 5, the leaves would be 
        # in nodes 4 to 8. If 6, then nodes 5 to 10.
        self.tree = np.zeros(2 * capacity - 1)
        # This buffer holds the data
        self.data = np.zeros(capacity, dtype=object)
        self.allocated = 0
        self.prev_push = None
        self.next_push = 0
        self.first_leaf = capacity - 1

    def push(self, data, weight):
        self.prev_push = self.next_push
        self.data[self.prev_push] = data
        self.set_weight(self.prev_push, weight)

        if (self.allocated + 1) < self.capacity:
            self.allocated += 1
            self.next_push += 1
        else:
            self.next_push = (self.next_push + 1) % self.capacity

    def get_weight(self, index):
        leaf = index + self.first_leaf
        return self.tree[leaf]

    def get_data(self, index):
        return self.data[index]

    def get_total_weight(self):
        return self.tree[0]

    def get_average_weight(self):
        return self.tree[0] / self.allocated

    def set_weight(self, index, weight):
        leaf = index + self.first_leaf
        change = weight - self.tree[leaf]
        self._change_weight_recursive(leaf, change)

    def _change_weight_recursive(self, node, change):
        self.tree[node] += change
        if node > 0:
            parent = (node-1) // 2
            self._change_weight_recursive(parent, change)


    def get_index_by_weight(self, weight, parent=0):
        if parent >= self.first_leaf:
            return parent - self.first_leaf

        left_child = parent*2 + 1
        right_child = parent*2 + 2

        if weight <= self.tree[left_child]:
            return self.get_index_by_weight(weight, left_child)
        else:
            return self.get_index_by_weight(weight-self.tree[left_child], right_child)

