from sympy import *
import re
from sympy.parsing.sympy_parser import parse_expr
from sympy.solvers import solve
from sympy.printing.printer import Printer

Printer.set_global_settings(num_columns=2000)
precision = 5


def globalExprs(*elementExprs):
    globalExprs = {}
    for elementExpr in elementExprs:
        for node, expr in elementExpr.items():
            globalExprs[node] = expr + \
                (globalExprs[node] if node in globalExprs else 0)
    return globalExprs


def elementExprs(exprs, nodes):
    return {nodes[n]: exprs[n] for n in range(len(nodes))}


def conductanceMatrix(kv, Av, lv, hv, pv, hvEnd=0):
    return kv * Av / lv * \
        Matrix([[1, -1], [-1, 1]]) + hv * pv * \
        lv / 6 * Matrix([[2, 1], [1, 2]]) + Matrix([[0, 0], [0, hvEnd*Av]])


def loadMatrix(hv, pv, lv, Tf, hvEnd=0, Av=0):
    return hv * pv * lv * Tf / 2 * Matrix([[1], [1]]) + Matrix([[0], [hvEnd * Av * Tf]])


def ThermalExprs(kv, Av, lv, hv, pv, Tf, ni, nj, hvEnd=0):
    cMatrix = conductanceMatrix(kv, Av, lv, hv, pv, hvEnd)
    TMatrix = Matrix([[Symbol("T" + str(ni))], [Symbol("T" + str(nj))]])
    lMatrix = loadMatrix(hv, pv, lv, Tf, hvEnd, Av)
    return cMatrix * TMatrix - lMatrix


def ThermalElementExprs(kv, Av, lv, hv, pv, Tf, ni, nj, hvEnd=0):
    return elementExprs(ThermalExprs(kv, Av, lv, hv, pv, Tf, ni, nj, hvEnd), [Symbol("T" + str(ni)), Symbol("T" + str(nj))])


def ThermalGlobalExprs(*dicts):
    return globalExprs(*[ThermalElementExprs(**i) for i in dicts])

def gaussianElimination(A, y, x, precision = 5):
    for i in range(0, A.shape[0]):
        Aii = A[i, i]
        for j in reversed(range(i + 1, A.shape[0])):
            if A[j, i] == 0:
                continue
            print("(Eq" + str(j+1) + ")-(" +
                  str(A[j, i]/A[i, i]) + ")(Eq" + str(i+1) + "):")
            y[j] = N(y[j] - y[i]*A[j, i]/A[i, i], precision)
            for k in reversed(range(i, A.shape[1])):
                A[j, k] = N(A[j, k] - A[i, k]*A[j, i]/A[i, i], precision)
            A[j, i] = 0
            pprint((A, y))
            print('\n\n')
    
    print('Backward substitution: \n')
    Ax = A * x
    symbols = {symbol[0]:symbol[0] for symbol in x.tolist()}
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

def solveThermalProblem(*dicts, boundaryConditions, precision = 5):
    thermalElementExprs = [ThermalElementExprs(**i) for i in dicts]
    print("The Conductance Matrix (K) and Load Matrix (F) are: \n")
    for i, thermalElementExpr in enumerate(thermalElementExprs):
        equations = list(thermalElementExpr.values())
        symbols = list(thermalElementExpr.keys())
        K, F = linear_eq_to_matrix(equations, symbols)
        pprint(Eq(Symbol('[K]') ** Symbol('(' + str(i + 1) + ')'), K, evaluate=False))
        print('\n')
        pprint(Eq(Symbol('[F]') ** Symbol('(' + str(i + 1) + ')'), F, evaluate=False))
        print('\n')
        print('\n')

    thermalGlobalExprs = globalExprs(*thermalElementExprs)
    print("The global Conductance Matrix (K) and Load Matrix (F): \n")
    equations = list(thermalGlobalExprs.values())
    symbols = list(thermalGlobalExprs.keys())
    K, F = linear_eq_to_matrix(equations, symbols)
    pprint(Eq(Symbol('[K]') ** Symbol('(G)'), K, evaluate=False))
    print('\n')
    pprint(Eq(Symbol('[F]') ** Symbol('(G)'), F, evaluate=False))
    print('\n')
    print('\n')

    print("Applying the boundary condition: \n")
    equations = list(thermalGlobalExprs.values())
    symbols = list(thermalGlobalExprs.keys())
    for k, v in boundaryConditions.items():
        boundaryConditionEq = Eq(Symbol(k), v)
        node = int(re.search('[0-9]+', k).group())
        equations[node - 1] = boundaryConditionEq
    K, F = linear_eq_to_matrix(equations, symbols)
    TMatrix = Matrix([[symbol] for symbol in symbols])
    pprint(Eq(UnevaluatedExpr(K) * UnevaluatedExpr(TMatrix), F, evaluate=False))
    print('\n')
    print('\n')

    print("Simplify the matrix equation: \n")
    equations = list(thermalGlobalExprs.values())
    symbols = list(thermalGlobalExprs.keys())
    for k, v in boundaryConditions.items():
        node = int(re.search('[0-9]+', k).group())
        equations.remove(equations[node - 1])
        symbols.remove(symbols[node - 1])
        equations = [eq.subs({Symbol(k): v}) for eq in equations]
    K, F = linear_eq_to_matrix(equations, symbols)
    TMatrix = Matrix([[symbol] for symbol in symbols])
    pprint(Eq(UnevaluatedExpr(K) * UnevaluatedExpr(TMatrix), F, evaluate=False))
    print('\n')
    print('\n')

    print("Apply Gaussian Elimination Method: \n")
    TResult = gaussianElimination(K, F, TMatrix, precision)

    print("The Temperature are: ")
    TResult = {**TResult, **{Symbol(k): v for k,v in boundaryConditions.items()}}
    pprint(TResult)

    print('Calculate the heat loss: ')
    QTotal = 0
    for i, obj in enumerate(dicts):
        hv = UnevaluatedExpr(obj['hv'])
        pv = UnevaluatedExpr(obj['pv'])
        lv = UnevaluatedExpr(obj['lv'])
        Ti = UnevaluatedExpr(Symbol('T' + str(obj['ni'])).subs(TResult))
        Tj = UnevaluatedExpr(Symbol('T' + str(obj['nj'])).subs(TResult))
        Tf = UnevaluatedExpr(obj['Tf'])
        Q = hv * pv * lv * ((Ti + Tj)/2 - Tf)
        # pprint(Eq(Symbol('Q')**Symbol('(' + str(i+1) + ')'),Q, evaluate = False))
        pprint(Eq(Symbol('Q')**Symbol('(' + str(i+1) + ')'),N(Q.doit(), precision)))
        QTotal += Q.doit()
        print()
    pprint(Eq(Symbol('Q_total'),N(QTotal, precision)))
    

# solveThermalProblem(
#     {"ni": 1, "nj": 2, "kv": 168, "Av": 5e-6, "lv": 20e-3, "hv": 30, "pv": 12e-3, "Tf": 20},
#     {"ni": 2, "nj": 3, "kv": 168, "Av": 5e-6, "lv": 20e-3, "hv": 30, "pv": 12e-3, "Tf": 20},
#     {"ni": 3, "nj": 4, "kv": 168, "Av": 5e-6, "lv": 20e-3, "hv": 30, "pv": 12e-3, "Tf": 20},
#     {"ni": 4, "nj": 5, "kv": 168, "Av": 5e-6, "lv": 20e-3, "hv": 30, "pv": 12e-3, "Tf": 20},
#     boundaryConditions={'T1': 100},
#     precision=4
# )

solveThermalProblem(
    {"ni": 1, "nj": 2, "kv": 0.08,  "Av": 1, "lv": 0.05, "hv": 0, "pv": 0, "Tf": 30},
    {"ni": 2, "nj": 3, "kv": 0.074, "Av": 1, "lv": 0.15, "hv": 0, "pv": 0, "Tf": 30},
    {"ni": 3, "nj": 4, "kv": 0.72,  "Av": 1, "lv": 0.1,  "hv": 0, "pv": 0, "Tf": 30, "hvEnd": 40},
    boundaryConditions={'T1': 200},
    precision=8
)