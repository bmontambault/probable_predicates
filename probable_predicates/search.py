import numpy as np
import itertools

from .predicate import Predicate

class Search(object):

    def __init__(self, data, dtypes, val_col, model):
        self.data = data
        self.dtypes = dtypes
        self.val_col = val_col
        self.model = model
        self.loss = self.model.score(data)

    def get_base_predicates(self, binsize=20):
        base_predicates = []
        for col in self.data.columns:
            if col != self.val_col:
                if self.dtypes[col] == 'discrete':
                    dtypes = {col: 'discrete'}
                    for val in self.data[col].unique():
                        base_predicates.append(Predicate({col: [val]}, dtypes))
                elif self.dtypes[col] == 'continuous':
                    dtypes = {col: 'continuous'}
                    vals = self.data[col].sort_values()
                    bins = np.array_split(vals, len(vals)/binsize)
                    for i in range(len(bins)):
                        lower = bins[i].iloc[0]
                        if i >= len(bins) - 1:
                            upper = bins[i].iloc[-1]
                        else:
                            upper = bins[i + 1].iloc[0]
                        base_predicates.append(Predicate({col: [[lower, upper]]}, dtypes))
        return base_predicates

    def score_predicate(self, predicate, penalty=1.):

        old_loss = self.loss[self.data.index].mean()
        new_data = self.data.query('~(' + predicate.query() + ')')
        new_loss = self.loss[new_data.index].mean()

        n = (self.data.shape[0] - new_data.shape[0])
        if n > 0:
            return (old_loss - new_loss) / n**penalty
        else:
            return 0

class BottomUp(Search):

    def __init__(self, data, dtypes, val_col, model):
        self.data = data
        self.dtypes = dtypes
        self.val_col = val_col
        self.model = model
        self.loss = self.model.score(data)

    def intersect(self, predicates):
        intersected = [p1.merge(p2) for p1, p2 in itertools.combinations(predicates, 2) if p1.fields != p2.fields]
        return intersected + predicates

    def max_predicate_score(self, predicate):
        components = [Predicate({k:v}, {k:predicate.dtypes[k]}) for k,v in predicate.fields.items()]
        return np.max([self.score_predicate(p) for p in components])

    def prune(self, predicates, scores, best_score):
        ret = [i for i in range(len(predicates)) if scores[i] < best_score]
        ret = [i for i in ret if self.max_predicate_score(predicates[i]) < best_score]
        return np.array(predicates)[ret], np.array(scores)[ret]

    def merge_adjacent(self, predicate, score, predicates, penalty, eps):
        adjacent = [p for p in predicates if predicate.is_adjacent(p)]
        for p in adjacent:
            merged_p = predicate.merge(p)
            merged_score = self.score_predicate(merged_p, penalty)
            if merged_score > score - eps:
                predicates = [pred for pred in predicates if pred.fields != p.fields]
                predicate, score, predicates = self.merge_adjacent(merged_p, merged_score, predicates, penalty, eps)
        return predicate, score, predicates

    def merge(self, predicates, scores, penalty, eps):
        merged = []
        merged_scores = []
        while len(predicates) > 0:
            predicate = predicates.pop(0)
            score = scores.pop(0)
            predicate, score, predicates = self.merge_adjacent(predicate, score, predicates, penalty, eps)
            merged.append(predicate)
            merged_scores.append(score)
        return merged, merged_scores

    def search(self, binsize=20, penalty=1., eps=.00001, maxiters=5):
        predicates = None
        best = None
        best_score = -np.inf
        base_predicates = self.get_base_predicates(binsize)
        for i in range(maxiters):
            if predicates is None:
                predicates = base_predicates.copy()
            else:
                predicates = [best.merge(p) for p in base_predicates if list(p.fields.keys())[0] not in best.fields]

            scores = [self.score_predicate(p, penalty) for p in predicates]
            sorted_scores = np.sort(scores)[::-1].tolist()
            sorted_predicates = np.array(predicates)[np.argsort(scores)][::-1].tolist()
            predicates, merged_scores = self.merge(sorted_predicates, sorted_scores, penalty, eps)
            best_merged_score = np.max(merged_scores)
            best_merged = predicates[np.argmax(merged_scores)]

            if best_merged_score > best_score:
                best_score = best_merged_score
                best = best_merged
            else:
                return best
        return best