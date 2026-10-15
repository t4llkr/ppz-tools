# -*- coding: utf-8 -*-
"""
ppz_sat.py — проверка единственности решения PathPix через SAT (python-sat).

Кодировка: булева переменная на каждое ребро между одноцветными соседями;
степени клеток — кардинальностные ограничения (==1 на концах, ==2 внутри).
Связность и точные длины путей — лениво: каждая найденная модель проверяется,
невалидные компоненты блокируются клозами. API совместим с ppz_gen.count_solutions.

Установка:  pip install python-sat
Самотест:   python ppz_sat.py
"""
import time
from collections import deque

DIRS4 = ((0, 1), (0, -1), (1, 0), (-1, 0))


def _components(chosen):
    """chosen: {cell: set(cells)} -> список компонент (каждая — список клеток)"""
    seen = set()
    comps = []
    for rc in chosen:
        if rc in seen:
            continue
        comp = []
        stack = [rc]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.append(x)
            stack.extend(chosen[x])
        comps.append(comp)
    return comps


def count_solutions(cells, rows, cols, limit=2, time_budget=60.0,
                    prefer=None, max_iters=1000, verbose=False,
                    deadline=None, **_ignored):
    """-> (count, sols, aborted); sols — списки множеств frozenset-рёбер."""
    from pysat.solvers import Solver
    from pysat.card import CardEnc, EncType

    neigh = {}
    for (r, c), cl in cells.items():
        if cl['clue'] == 1:
            neigh[(r, c)] = []
            continue
        neigh[(r, c)] = [(r + dr, c + dc) for dr, dc in DIRS4
                         if (r + dr, c + dc) in cells
                         and cells[(r + dr, c + dc)]['color'] == cl['color']
                         and cells[(r + dr, c + dc)]['clue'] != 1]
    req = {rc: (0 if cl['clue'] == 1 else (1 if cl['clue'] > 1 else 2))
           for rc, cl in cells.items()}

    solver = Solver(name='g4')
    var = {}
    top = [0]

    def newvar():
        top[0] += 1
        return top[0]

    edges = []
    for rc, ns in neigh.items():
        for x in ns:
            key = frozenset((rc, x))
            if key in var:
                continue
            var[key] = newvar()
            edges.append(key)
    for key, v in var.items():
        a, b = tuple(key)
        # стены запрещают ребро
        ar, ac = a
        dr, dc = b[0] - ar, b[1] - ac
        WBIT = {(0, -1): 1, (-1, 0): 2, (0, 1): 4, (1, 0): 8}
        if cells[a]['wall'] & WBIT[(dr, dc)]:
            solver.add_clause([-v])
        if cells[b]['wall'] & WBIT[(-dr, -dc)]:
            solver.add_clause([-v])

    for rc, ns in neigh.items():
        if req[rc] == 0:
            for x in ns:
                solver.add_clause([-var[frozenset((rc, x))]])
            continue
        lits = [var[frozenset((rc, x))] for x in ns]
        if len(lits) == req[rc]:
            for v in lits:
                solver.add_clause([v])
        elif len(lits) > req[rc]:
            cnf = CardEnc.equals(lits, bound=req[rc], top_id=top[0],
                                 encoding=EncType.seqcounter)
            top[0] = cnf.nv
            for cl_ in cnf.clauses:
                solver.add_clause(cl_)

    if prefer:
        solver.set_phases([var[e] for e in var if e in prefer])

    def check(model_edges):
        """проверка компонент; -> (valid, [invalid_edge_sets])"""
        chosen = {rc: set() for rc in cells}
        for e in model_edges:
            a, b = tuple(e)
            chosen[a].add(b)
            chosen[b].add(a)
        bad = []
        for comp in _components({rc: s for rc, s in chosen.items() if req[rc] > 0}):
            ends = [x for x in comp if req[x] == 1]
            ok = len(ends) == 2
            if ok:
                a, b = cells[ends[0]], cells[ends[1]]
                ok = (a['color'] == b['color'] and a['clue'] == b['clue']
                      and len(comp) == a['clue'])
            if ok:
                # цепочка без ветвей: обход от ends[0]
                path = [ends[0]]
                cur = next(iter(chosen[ends[0]]))
                prev = ends[0]
                while cur != ends[1]:
                    if req[cur] != 2 or len(chosen[cur]) != 2:
                        ok = False
                        break
                    nxt = [x for x in chosen[cur] if x != prev][0]
                    if nxt in path:
                        ok = False
                        break
                    path.append(cur)
                    prev, cur = cur, nxt
                    if len(path) > len(comp):
                        ok = False
                        break
                if ok:
                    path.append(ends[1])
                    ok = len(path) == len(comp)
            if not ok:
                bad.append({var[frozenset((comp[i], comp[j]))]
                            for i in range(len(comp))
                            for j in range(i + 1, len(comp))
                            if comp[j] in chosen[comp[i]]})
        return (not bad), bad

    import threading
    t0 = time.time()
    if deadline is not None:
        time_budget = min(time_budget, max(0.1, deadline - t0))
    count, sols = 0, []
    aborted = False
    for it in range(max_iters):
        if time.time() - t0 > time_budget:
            aborted = True
            break
        # жёсткий лимит на один вызов solve: конфликт-бюджет + прерывание по таймеру
        try:
            solver.conf_budget = 20000
        except Exception:
            pass
        timer = None
        if time_budget < 1e8:
            timer = threading.Timer(max(0.05, time_budget - (time.time() - t0)),
                                    solver.interrupt)
            timer.start()
        res = solver.solve_limited(expect_interrupt=True)
        if timer is not None:
            timer.cancel()
        if res is None:
            aborted = True
            break
        if not res:
            break
        model = set(v for v in solver.get_model() if v > 0)
        true_edges = {e for e, v in var.items() if v in model}
        valid, bad = check(true_edges)
        if valid:
            count += 1
            sols.append(true_edges)
            solver.add_clause([-var[e] for e in true_edges])
            if count >= limit:
                break
        else:
            for comp_vars in bad:
                solver.add_clause([-v for v in comp_vars])
        if verbose and it % 50 == 0:
            print(f'  sat: итерация {it}, решений {count}, {time.time()-t0:.1f} с')
    solver.delete()
    return count, sols, aborted


def _selftest():
    line = {(0, c): dict(color=1, clue=4 if c in (0, 3) else 0, hi=0, lo=0, wall=0)
            for c in range(4)}
    print('1x4:', count_solutions(line, 1, 4), '(ожид. count=1)')
    amb = {}
    for r in range(2):
        for c in range(3):
            amb[(r, c)] = dict(color=1, clue=3 if (r, c) in ((0, 0), (1, 0), (0, 2), (1, 2)) else 0,
                               hi=0, lo=0, wall=0)
    print('2x3:', count_solutions(amb, 2, 3), '(ожид. count=1)')
    ring = {}
    for r in range(2):
        for c in range(3):
            ring[(r, c)] = dict(color=1, clue=6 if (r, c) in ((0, 0), (1, 2)) else 0,
                                hi=0, lo=0, wall=0)
    print('2x3 пара 6:', count_solutions(ring, 2, 3), '(ожид. count>=1)')


if __name__ == '__main__':
    _selftest()
