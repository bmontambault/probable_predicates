class Predicate:

    def __init__(self, fields, dtypes):
        self.fields = fields
        self.dtypes = dtypes

    def is_adjacent_discrete(self, key, p):
        return key in p.fields

    def is_adjacent_continuous_val(self, val1, val2):
        return (val1[0] >= val2[0] and val1[1] <= val2[1]) \
            or (val2[0] >= val1[0] and val2[1] <= val1[1]) \
            or (val2[0] <= val1[1] and val2[1] >= val1[0]) \
            or (val1[0] <= val2[1] and val1[1] >= val2[0])

    def is_adjacent_continuous(self, key, p):
        if key in p.fields:
            vals1 = self.fields[key]
            vals2 = p.fields[key]
            for i in range(len(vals1)):
                val1 = vals1[i]
                for j in range(len(vals2)):
                    val2 = vals2[j]
                    if self.is_adjacent_continuous_val(val1, val2):
                        return True
        return False

    def is_adjacent(self, p):
        for k in self.fields.keys():
            if k not in p.fields.keys():
                return False
            else:
                if self.dtypes[k] == 'discrete':
                    is_adj = self.is_adjacent_discrete(k, p)
                elif self.dtypes[k] == 'continuous':
                    is_adj = self.is_adjacent_continuous(k, p)
                if not is_adj:
                    return False
        return True

    def merge_discrete_field(self, key, p):
        return list(set(self.fields[key] + p.fields[key]))

    def merge_continuous_field_val(self, val1, val2):
        if val1[1] >= val2[1]:
            left = val2
            right = val1
        elif val2[1] >= val1[1]:
            left = val1
            right = val2
        if right[0] <= left[0]:
            new_val = right
        elif right[0] <= left[1]:
            new_val = [left[0], right[1]]
        return new_val

    def merge_continuous_field(self, key, p):
        vals1 = self.fields[key]
        vals2 = p.fields[key]
        merged_idx = []
        merged = []
        for i in range(len(vals1)):
            val1 = vals1[i]
            for j in range(len(vals2)):
                val2 = vals2[j]
                if self.is_adjacent_continuous_val(val1, val2):
                    merged_idx.append(j)
                    val1 = self.merge_continuous_field_val(val1, val2)
            merged.append(val1)

        for i in range(len(vals2)):
            if i not in merged_idx:
                merged.append(vals2[i])
        return merged

    def merge_field(self, key, p):
        if self.dtypes[key] == 'discrete':
            return self.merge_discrete_field(key, p)
        elif self.dtypes[key] == 'continuous':
            return self.merge_continuous_field(key, p)

    def merge(self, p):
        new_fields = self.fields.copy()
        new_dtypes = self.dtypes.copy()
        for k in p.fields.keys():
            if k in new_fields:
                new_fields[k] = self.merge_field(k, p)
            else:
                new_fields[k] = p.fields[k]
                new_dtypes[k] = p.dtypes[k]
        return Predicate(new_fields, new_dtypes)

    def query_continuous(self, feature):
        return f"({' or '.join([f'({feature} >= {val[0]} and {feature} <= {val[1]})' for val in self.fields[feature]])})"

    def query_discrete(self, feature):
        return f'({feature} in {self.fields[feature]})'

    def query(self):
        p = [self.query_discrete(feature) if self.dtypes[feature] == 'discrete' else self.query_continuous(feature)
           for feature in self.fields.keys()]
        return ' and '.join(p)
