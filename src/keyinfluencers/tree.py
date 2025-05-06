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


class DecisionTreeInsight:

    def __init__(self, rule, segment_metric, overall_metric, metric, lift_pct, sample_size, target):
        self.rule = rule
        self.segment_metric = segment_metric
        self.overall_metric = overall_metric
        self.metric = metric
        self.lift_pct = lift_pct
        self.sample_size = sample_size
        self.target = target

    def __repr__(self):
        direction = "increases" if self.lift_pct > 0 else "decreases"
        return (f"- When {self.rule}, the {self.metric} of '{self.target}' {direction} by {abs(self.lift_pct):.1f}% "
              f"(segment = {self.segment_metric:.2f}, overall = {self.overall_metric:.2f}, n = {int(self.sample_size)})")
    
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


def extract_tree_insights(tree, feature_names, overall_metric, tree_type, top_n=5, target=None, focus_class_index=None, focus_class=None):
    tree_ = tree.tree_
    insights = []

    def traverse_tree(node, path):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]
            traverse_tree(tree_.children_left[node], path + [f"{name} <= {threshold:.2f}"])
            traverse_tree(tree_.children_right[node], path + [f"{name} > {threshold:.2f}"])
        else:
            samples = tree_.n_node_samples[node]
            value = tree_.value[node][0]
            segment_metric = value[0] / samples if tree_type == 'regression' else value[focus_class_index] / value.sum()
            lift = (segment_metric - overall_metric) / overall_metric
            insights.append(DecisionTreeInsight(
                rule=' AND '.join(path),
                segment_metric=segment_metric,
                overall_metric=overall_metric,
                metric = "likelihood" if tree_type == 'classification' else "value",
                lift_pct=lift * 100,
                sample_size=samples,
                target=focus_class if focus_class is not None else target
            ))
    
    traverse_tree(0, [])

    insights.sort(key=lambda x: x.lift_pct, reverse=True)
    if top_n is not None:
        insights = insights[:top_n]
    return insights