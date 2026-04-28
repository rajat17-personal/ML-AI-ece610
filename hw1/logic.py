class Expr:
    def __init__(self, op, *args):
        # If args is empty, op is a symbol; otherwise, op is an operator with args
        self.op = op
        self.args = args

    def __and__(self, other):
        return Expr('AND', self, other)

    def __or__(self, other):
        return Expr('OR', self, other)

    def __invert__(self):
        return Expr('NOT', self)

    def __rshift__(self, other):
        return Expr('IMPLIES', self, other)

    def __mod__(self, other):
        return Expr('BICOND', self, other)

    def __repr__(self):
        # Pretty-print
        if not self.args:
            return self.op
        if self.op == 'NOT':
            return f"¬{self.args[0]}"
        if self.op in ('AND', 'OR', 'IMPLIES', 'BICOND'):
            symbol = {'AND':'∧', 'OR':'∨', 'IMPLIES':'→', 'BICOND':'↔'}[self.op]
            left, right = self.args
            return f"({left} {symbol} {right})"
        joined = ', '.join(map(str, self.args))
        return f"{self.op}({joined})"

    def __eq__(self, other):
        return isinstance(other, Expr) and self.op == other.op and self.args == other.args

    def __hash__(self):
        return hash((self.op, self.args))

def And(*args):
    if len(args) == 0:
        raise ValueError("And() needs at least one argument")
    result = args[0]
    for a in args[1:]:
        result = result & a
    return result


def Or(*args):
    if len(args) == 0:
        raise ValueError("Or() needs at least one argument")
    result = args[0]
    for a in args[1:]:
        result = result | a
    return result


def Not(arg):
    return ~arg

# CNF conversion routines

def eliminate_implications(e):
    if e.op == 'BICOND':
        A, B = e.args
        return eliminate_implications((A >> B) & (B >> A))
    if e.op == 'IMPLIES':
        A, B = e.args
        return eliminate_implications(Not(A) | B)
    if not e.args:
        return e
    return Expr(e.op, *[eliminate_implications(arg) for arg in e.args])


def move_not_inwards(e):
    if e.op == 'NOT':
        sub = e.args[0]
        if sub.op == 'NOT':
            return move_not_inwards(sub.args[0])
        if sub.op == 'AND':
            return Or(*[move_not_inwards(Not(arg)) for arg in sub.args])
        if sub.op == 'OR':
            return And(*[move_not_inwards(Not(arg)) for arg in sub.args])
        return e
    if not e.args:
        return e
    return Expr(e.op, *[move_not_inwards(arg) for arg in e.args])


def distribute_or_over_and(e):
    if e.op == 'OR':
        A, B = [distribute_or_over_and(arg) for arg in e.args]
        if A.op == 'AND':
            return And(*[distribute_or_over_and(Or(arg, B)) for arg in A.args])
        if B.op == 'AND':
            return And(*[distribute_or_over_and(Or(A, arg)) for arg in B.args])
        return Or(A, B)
    if not e.args:
        return e
    return Expr(e.op, *[distribute_or_over_and(arg) for arg in e.args])


def to_cnf(e):
    step1 = eliminate_implications(e)
    step2 = move_not_inwards(step1)
    step3 = distribute_or_over_and(step2)
    return step3

# Helper for Question 2: Exactly one true

def ExactOne(vars):
    # At least one is true
    at_least = Or(*vars)
    # At most one is true: pairwise no two can both be true
    at_most = And(*[
        Or(Not(v1), Not(v2))
        for i, v1 in enumerate(vars)
        for v2 in vars[i+1:]
    ])
    return And(at_least, at_most)
