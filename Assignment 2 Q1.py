from sympy import *
import re
from sympy.parsing.sympy_parser import parse_expr
from sympy.solvers import solve
from sympy.printing.printer import Printer

Printer.set_global_settings(num_columns=2000)
precision = 5


def globalExprs(*elementExprs, precision=5):
    globalExprs = {}
    for elementExpr in elementExprs:
        for node, expr in elementExpr.items():
            globalExprs[node] = N(expr +
                                  (globalExprs[node] if node in globalExprs else 0), precision)
    return globalExprs


def elementExprs(exprs, nodes):
    return {nodes[n]: exprs[n] for n in range(len(nodes))}


def stiffnessMatrix(ni, nj, area, Ev, length, angle):
    angle = rad(angle)
    return area * Ev / length * Matrix([
        [cos(angle)**2, sin(angle)*cos(angle), -
         cos(angle)**2, -sin(angle)*cos(angle)],
        [sin(angle)*cos(angle), sin(angle)**2, -
         sin(angle)*cos(angle), -sin(angle)**2],
        [-cos(angle)**2, -sin(angle)*cos(angle),
         cos(angle)**2, sin(angle)*cos(angle)],
        [-sin(angle)*cos(angle), -sin(angle)**2,
         sin(angle)*cos(angle), sin(angle)**2],
    ])


def TrussesExprs(ni, nj, area, Ev, length, angle):
    UMatrix = Matrix([
        [Symbol("U" + str(ni) + "x")],
        [Symbol("U" + str(ni) + "y")],
        [Symbol("U" + str(nj) + "x")],
        [Symbol("U" + str(nj) + "y")],
    ])
    sMatrix = stiffnessMatrix(ni, nj, area, Ev, length, angle)
    return sMatrix * UMatrix


def TrussesElementExprs(ni, nj, area, Ev, length, angle):
    return elementExprs(TrussesExprs(ni, nj, area, Ev, length, angle), [
        Symbol("U" + str(ni) + "x"),
        Symbol("U" + str(ni) + "y"),
        Symbol("U" + str(nj) + "x"),
        Symbol("U" + str(nj) + "y"),
    ])


def gaussianElimination(A, y, x, precision=5):
    for i in range(0, A.shape[0]):
        Aii = A[i, i]
        for j in reversed(range(i + 1, A.shape[0])):
            if A[j, i] == 0:
                continue
            print("(Eq" + str(j+1) + ")-(" +
                  str(A[j, i]/A[i, i]) + ")(Eq" + str(i+1) + "):")
            y[j] = N((y[j] - y[i] * A[j, i]/A[i, i]).evalf(), precision)
            for k in reversed(range(i, A.shape[1])):
                A[j, k] = N(A[j, k] - A[i, k]*A[j, i]/A[i, i], precision)
            A[j, i] = 0
            pprint((A, y))
            print('\n\n')

    print('Backward substitution: \n')
    Ax = A * x
    symbols = {symbol[0]: symbol[0] for symbol in x.tolist()}
    for i in reversed(range(x.rows)):
        lExpr = Ax[i]
        pprint(Eq(lExpr, y[i], evaluate=False))
        if(len(lExpr.free_symbols) > 1):
            for k, v in symbols.items():
                with evaluate(False):
                    lExpr = lExpr.replace(k, v)
            pprint(Eq(lExpr, y[i], evaluate=False))
        solveResult = solve(lExpr - y[i], dict=True)[0]
        for k, v in solveResult.items():
            v = N(v, precision)
            pprint(Eq(k, v))
            symbols[k] = v

        print('\n')
    return symbols


def solveTrussesProblem(*dicts, boundaryConditions, area, Ev, precision=5):
    thermalElementExprs = [TrussesElementExprs(
        **i, **{"area": area, "Ev": Ev}) for i in dicts]
    print("The Conductance Matrix (K) are: \n")
    for i, thermalElementExpr in enumerate(thermalElementExprs):
        equations = list(thermalElementExpr.values())
        symbols = list(thermalElementExpr.keys())
        K, F = linear_eq_to_matrix(equations, symbols)
        K = N(K, precision)
        F = N(F, precision)
        pprint(
            Eq(Symbol('[K]') ** Symbol('(' + str(i + 1) + ')'), K, evaluate=False))
        print('\n')
        print('\n')

    thermalGlobalExprs = globalExprs(*thermalElementExprs, precision=precision)
    thermalGlobalExprs = {
        k: v - Symbol(str(k).replace('U', 'f')) for k, v in thermalGlobalExprs.items()}
    print("The global Conductance Matrix (K): \n")
    symbols = list(thermalGlobalExprs.keys())
    equations = list(thermalGlobalExprs.values())
    K, F = linear_eq_to_matrix(equations, symbols)
    K = N(K, precision)
    F = N(F, precision)
    pprint(Eq(Symbol('[K]') ** Symbol('(G)'), K, evaluate=False))
    print('\n')
    print('\n')

    print("Applying the boundary condition: \n")
    equations = list(thermalGlobalExprs.values())
    symbols = list(thermalGlobalExprs.keys())
    for k, v in boundaryConditions.items():
        boundaryConditionEq = Eq(Symbol(k), v)
        if Symbol(k) in symbols:
            node = symbols.index(Symbol(k))
            equations[node] = boundaryConditionEq
        else:
            equations = [equation.subs({Symbol(k): v})
                         for equation in equations]
    K, F = linear_eq_to_matrix(equations, symbols)
    K = N(K, precision)
    F = N(F, precision)
    UMatrix = Matrix([[symbol] for symbol in symbols])
    pprint(Eq(UnevaluatedExpr(K) * UnevaluatedExpr(UMatrix), F, evaluate=False))
    print('\n')
    print('\n')

    print("Simplify the matrix equation: \n")
    equations = list(thermalGlobalExprs.values())
    symbols = list(thermalGlobalExprs.keys())
    for k, v in boundaryConditions.items():
        boundaryConditionEq = Eq(Symbol(k), v)
        equations = [equation.subs({Symbol(k): v})
                     for equation in equations]
        if Symbol(k) in symbols:
            node = symbols.index(Symbol(k))
            symbols.remove(symbols[node])
            equations.remove(equations[node])
    equations = [N(equation, precision) for equation in equations]
    K, F = linear_eq_to_matrix(equations, symbols)
    UMatrix = Matrix([[symbol] for symbol in symbols])
    pprint(Eq(UnevaluatedExpr(K) * UnevaluatedExpr(UMatrix), F, evaluate=False))
    print('\n')
    print('\n')

    print("Apply Gaussian Elimination Method: \n")
    UResult = gaussianElimination(K, F, UMatrix, precision)

    print("The U are: ")
    UResult = {symbol: symbol.subs({**UResult, **{Symbol(k): v for k, v in boundaryConditions.items()}})
               for symbol in list(thermalGlobalExprs.keys())}
    pprint(UResult)
    print('\n')
    print('\n')

    print("The Reaction Force are:")
    equations = list(thermalGlobalExprs.values())
    symbols = list(thermalGlobalExprs.keys())
    KGlobal, F = linear_eq_to_matrix(equations, symbols)
    UMatrix = Matrix([[symbol.subs(UResult)] for symbol in symbols])
    FMatrix = F.subs({Symbol(k): v for k, v in boundaryConditions.items()})
    reactionForce = KGlobal * UMatrix - FMatrix
    pprint(reactionForce)
    print('\n')
    print('\n')

    print("the Normal Stress are:")
    for e, elem in enumerate(dicts):
        ni = elem['ni']
        nj = elem['nj']
        angle = rad(elem['angle'])
        length = elem['length']
        invTMatrix = N(Matrix([
            [cos(angle),  sin(angle), 0, 0],
            [-sin(angle), cos(angle), 0, 0],
            [0, 0, cos(angle),  sin(angle)],
            [0, 0, -sin(angle), cos(angle)],
        ]), precision)
        UMatrix = N(Matrix([
            [Symbol('U' + str(ni) + 'x')],
            [Symbol('U' + str(ni) + 'y')],
            [Symbol('U' + str(nj) + 'x')],
            [Symbol('U' + str(nj) + 'y')],
        ]).subs(UResult), precision)
        uMatrix = invTMatrix * UMatrix
        pprint(Eq(N(Matrix([
            [Symbol('u' + str(ni) + 'x')],
            [Symbol('u' + str(ni) + 'y')],
            [Symbol('u' + str(nj) + 'x')],
            [Symbol('u' + str(nj) + 'y')],
        ])), Eq(UnevaluatedExpr(invTMatrix) * UnevaluatedExpr(UMatrix), uMatrix, evaluate=False), evaluate=False))
        uix = uMatrix[0]
        ujx = uMatrix[2]
        sigma = N(Ev * (uix - ujx) / length, precision)
        pprint(Eq(Symbol('sigma^' + str(e)), sigma, evaluate=False))
        print('\n')
        print('\n')


solveTrussesProblem(
    {"ni": 1, "nj": 2, "length": sqrt(2), "angle": 45},
    {"ni": 1, "nj": 3, "length": 1,       "angle": 0},
    {"ni": 2, "nj": 3, "length": 1,       "angle": 90},
    {"ni": 2, "nj": 4, "length": 1,       "angle": 0},
    {"ni": 2, "nj": 5, "length": sqrt(2), "angle": -45},
    {"ni": 3, "nj": 5, "length": 1,       "angle": 0},
    {"ni": 4, "nj": 5, "length": 1,       "angle": 90},
    {"ni": 4, "nj": 6, "length": 1,       "angle": 0},
    {"ni": 5, "nj": 6, "length": sqrt(2), "angle": 45},
    {"ni": 5, "nj": 7, "length": 1,       "angle": 0},
    {"ni": 6, "nj": 7, "length": 1,       "angle": 90},
    {"ni": 6, "nj": 8, "length": sqrt(2), "angle": -45},
    {"ni": 7, "nj": 8, "length": 1,       "angle": 0},
    boundaryConditions={
        "U1x": 0,
        "U1y": 0,
        "U8x": 0,
        "U8y": 0,
        # forces
        "f1x": 0,
        "f1y": 0,
        "f2x": 0,
        "f2y": 0,
        "f3x": 0,
        "f3y": -30e3,
        "f4x": 0,
        "f4y": 0,
        "f5x": 0,
        "f5y": -30e3,
        "f6x": 0,
        "f6y": 0,
        "f7x": 0,
        "f7y": -30e3,
        "f8x": 0,
        "f8y": 0,
    },
    area=6e-2 * 6e-2,
    Ev=13.1e9,
    precision=4
)
