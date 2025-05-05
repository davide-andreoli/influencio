from sklearn.tree import _tree

class DecisionTreeRule:

    def __init__(self, feature, threshold, left, right, depth, impurity):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.depth = depth
        self.impurity = impurity

    def __repr__(self):
        return f"DecisionTreeRule(feature={self.feature}, threshold={self.threshold}, depth={self.depth}, impurity={self.impurity})"

    
def extract_tree_rules(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    rules = []

    def traverse_tree(node, depth):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            left_child = traverse_tree(tree_.children_left[node], depth + 1)
            right_child = traverse_tree(tree_.children_right[node], depth + 1)

            rule = DecisionTreeRule(name, threshold, left_child, right_child, depth, tree_.impurity[node])
            rules.append(rule)
            return rule
        else:
            return None

    traverse_tree(0, 0)
    return rules

def extract_feature_contributions(tree, feature_names):
    tree_ = tree.tree_
    
    feature_importances = tree_.compute_feature_importances()

    feature_contributions = [
        (feature_names[i], importance)
        for i, importance in enumerate(feature_importances)
    ]

    feature_contributions.sort(key=lambda x: x[1], reverse=True)

    return feature_contributions
