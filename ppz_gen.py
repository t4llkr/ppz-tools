#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ppz_gen.py — генератор уровней PathPix (.ppz) из изображений.

Стратегия единственности решения «по построению» (MVP):
  1. Картинка квантуется в палитру (<= 30 цветов + слот 0), каждая клетка обязана
     иметь цвет — фон картинки сам является набором путей.
  2. Каждая 4-связная монохромная область режется построчно на прямые «полосы»
     (коридоры шириной 1). Полоса длиной 1 -> особый квадратик (clue=1),
     длиной 2..31 -> путь (clue = длина на обоих концах),
     длиннее 31 -> режется на куски <= 31 (куски = отдельные пути).
  3. Коридор с фиксированными концами и длиной = его длине имеет РОВНО ОДИН
     гамильтонов путь (сам коридор) -> решение единственно, ЕСЛИ пути не могут
     обмениваться клетками. Обмен возможен только через общее ребро между путями
     ОДНОГО цвета: на каждое такое ребро ставится СТЕНА (с двух сторон).
     Разноцветные соседи обменяться не могут (путь монохромен).
  Итого: единственность гарантирована конструкцией, солвер не нужен.

CLI:
  python ppz_gen.py in.png out.ppz [--max 150] [--colors 10] [--preview prev.png]
"""
import argparse, struct, sys
from collections import Counter, deque
from PIL import Image, ImageDraw

# ---------- 1. изображение -> сетка ----------

def image_to_grid(img, max_size=150, n_colors=10, merge_thresh=24):
    """-> (grid[rows][cols] значения 1..K, palette [(a,r,g,b)...] длиной K+1, slot0=фон)"""
    img = img.convert('RGB')
    w, h = img.size
    s = min(1.0, max_size / max(w, h))
    if s < 1.0:
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    q = img.quantize(colors=max(2, min(30, n_colors)), method=Image.MEDIANCUT)
    idx = list(q.getdata())
    K = max(idx) + 1  # фактически использовано цветов (может быть < запрошенных)
    pal = q.getpalette()[:K * 3]
    # слияние визуально идентичных цветов: JPEG-шум раскалывает плоские заливки
    # на десятки близких индексов -> шахматные микро-области -> лавина квадратиков.
    # Порок мал vs типичное расстояние между настоящими цветами палитры.
    if merge_thresh > 0:
        parent = list(range(K))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def dist2(i, j):
            return (pal[i * 3] - pal[j * 3]) ** 2 + (pal[i * 3 + 1] - pal[j * 3 + 1]) ** 2 \
                   + (pal[i * 3 + 2] - pal[j * 3 + 2]) ** 2

        for i in range(K):
            for j in range(i + 1, K):
                if dist2(i, j) < merge_thresh ** 2:
                    parent[find(i)] = find(j)
        root_new = {}
        npal = []
        remap = [0] * K
        for i in range(K):
            r = find(i)
            if r not in root_new:
                root_new[r] = len(npal) // 3
                npal.extend(pal[r * 3:r * 3 + 3])
            remap[i] = root_new[r]
        idx = [remap[i] for i in idx]
        pal = npal
        K = len(npal) // 3
    cnt = Counter(idx)
    bg = cnt.most_common(1)[0][0]              # самый частый цвет = фон картинки
    order = [bg] + [i for i in range(K) if i != bg]
    remap = {old: new + 1 for new, old in enumerate(order)}   # 1..K
    rows, cols = q.size[1], q.size[0]
    grid = [[remap[idx[r * cols + c]] for c in range(cols)] for r in range(rows)]
    palette = [(255, pal[bg * 3], pal[bg * 3 + 1], pal[bg * 3 + 2])]  # слот 0 = фон (дубль)
    for old in order:  # все K цветов, включая фон: ячейки используют индексы 1..K
        palette.append((255, pal[old * 3], pal[old * 3 + 1], pal[old * 3 + 2]))
    return grid, palette

# ---------- 2. разбиение на пути ----------

MAX_PATH = 31
DIRS4 = ((0, 1), (0, -1), (1, 0), (-1, 0))

def decompose(grid, target=6, seed=0, bendiness=0.5):
    """Змейки внутри областей. target ~ желаемая длина пути (реальные уровни ~5),
    bendiness — вероятность предпочесть поворот продолжению прямой.
    -> (paths: [[(r,c),...]], squares: {(r,c)})"""
    import random
    rng = random.Random(seed)
    rows, cols = len(grid), len(grid[0])
    seen = [[False] * cols for _ in range(rows)]
    paths, squares, orphans = [], set(), []
    for r0 in range(rows):
        for c0 in range(cols):
            if seen[r0][c0]:
                continue
            color = grid[r0][c0]
            q = deque([(r0, c0)]); seen[r0][c0] = True; region = [(r0, c0)]
            while q:
                r, c = q.popleft()
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and not seen[nr][nc] and grid[nr][nc] == color:
                        seen[nr][nc] = True; q.append((nr, nc)); region.append((nr, nc))
            unvis = set(region)
            while unvis:
                # старт: клетка с наименьшим числом свободных соседей (Warnsdorff)
                def free_deg(cell):
                    rr, cc = cell
                    return sum((rr + dr, cc + dc) in unvis for dr, dc in DIRS4)
                start = min(unvis, key=lambda cell: (free_deg(cell), rng.random()))
                snake = [start]; unvis.discard(start)
                t_eff = max(2, min(MAX_PATH, round(target * (0.7 + 0.6 * rng.random()))))
                while len(snake) < t_eff:
                    r, c = snake[-1]
                    cands = [(r + dr, c + dc) for dr, dc in DIRS4 if (r + dr, c + dc) in unvis]
                    if not cands:
                        break
                    straight = None
                    if len(snake) >= 2:
                        pr, pc = snake[-2]
                        s = (r - pr, c - pc)
                        straight = (r + s[0], c + s[1])
                    pool = cands
                    if straight and straight in unvis and rng.random() > bendiness:
                        pool = [straight]          # продолжаем прямо
                    nxt = min(pool, key=lambda cell: (free_deg(cell), rng.random()))
                    snake.append(nxt); unvis.discard(nxt)
                if len(snake) == 1:
                    orphans.append(start)   # дожмём ниже, не всё станет квадратиками
                else:
                    paths.append(snake)
    # Дожим одиночек, три прохода:
    # 1) прикрепление к концам путей того же цвета (L-образный коридор —
    #    гамильтонова однозначность сохраняется);
    # 2) сцепка соседних одиночек в пути длины 2 (вставить одиночку МЕЖДУ
    #    клетками пути ортогонально нельзя — у ребра сетки нет общих соседей);
    # 3) повторное прикрепление — новые пары могут вырасти.
    def attach_ends():
        changed = False
        for o in list(orphans):
            r, c = o
            for path in paths:
                if len(path) >= MAX_PATH or grid[path[-1][0]][path[-1][1]] != grid[r][c]:
                    continue
                er, ec = path[-1]
                if abs(er - r) + abs(ec - c) == 1:
                    path.append(o); orphans.remove(o); changed = True
                    break
                er, ec = path[0]
                if abs(er - r) + abs(ec - c) == 1:
                    path.insert(0, o); orphans.remove(o); changed = True
                    break
        return changed

    def pair_orphans():
        changed = False
        for o in list(orphans):
            r, c = o
            for dr, dc in DIRS4:
                nb = (r + dr, c + dc)
                if nb in orphans and grid[nb[0]][nb[1]] == grid[r][c]:
                    paths.append([o, nb])
                    orphans.remove(o)
                    orphans.remove(nb)
                    changed = True
                    break
            if o not in orphans:
                break
        return changed

    while attach_ends():
        pass
    while pair_orphans():
        pass
    while attach_ends():
        pass
    squares = set(orphans)
    return paths, squares

# ---------- 2а. солвер единственности (подсчёт решений до limit) ----------

def count_solutions(cells, rows, cols, limit=2, forced=None, node_budget=200000,
                     deadline=None, progress=False, prefer=None):
    """cells: {(r,c): {color, clue}}. Возвращает (count, alt_edges или None).
    forced — iterable пар клеток, принудительно зафиксированных рёбер (каноническое решение);
    тогда ищутся решения, ОТЛИЧНЫЕ от forced. Стены соблюдаются."""
    neigh = {}
    for (r, c), cl in cells.items():
        if cl['clue'] == 1:
            neigh[(r, c)] = []
            continue
        ns = [(r + dr, c + dc) for dr, dc in DIRS4
              if (r + dr, c + dc) in cells and cells[(r + dr, c + dc)]['color'] == cl['color']
              and cells[(r + dr, c + dc)]['clue'] != 1]
        neigh[(r, c)] = ns
    req = {rc: (0 if cl['clue'] == 1 else (1 if cl['clue'] > 1 else 2)) for rc, cl in cells.items()}
    keys = list(cells)
    WBIT = {1: (0, -1), 2: (-1, 0), 4: (0, 1), 8: (1, 0)}
    import time as _t
    sol_count = 0
    sols = []
    nodes = [0]
    aborted = [False]
    t0 = _t.time()

    def full_check(chosen):
        """каждая компонента — цепочка между парой совпадающих концов без ветвей"""
        done = set()
        for rc in keys:
            if req[rc] == 0:
                done.add(rc)
                continue
            if rc in done:
                continue
            comp = []
            stack = [rc]
            while stack:
                x = stack.pop()
                if x not in comp:
                    comp.append(x)
                    stack.extend(chosen[x])
            ends = [x for x in comp if req[x] == 1]
            if len(ends) != 2:
                return False
            path = [ends[0]]
            cur = next(iter(chosen[ends[0]]))
            prev = ends[0]
            while cur != ends[1]:
                if req[cur] != 2 or len(chosen[cur]) != 2:
                    return False
                nxt = [x for x in chosen[cur] if x != prev][0]
                if nxt in path:
                    return False
                path.append(cur)
                prev, cur = cur, nxt
                if len(path) > len(comp):
                    return False
            path.append(ends[1])
            if len(path) != len(comp):
                return False
            a, b = cells[ends[0]], cells[ends[1]]
            if a['color'] != b['color'] or a['clue'] != b['clue'] or len(comp) != a['clue']:
                return False
            done.update(comp)
        return True

    def undo(log, base):
        while len(log) > base:
            op, a, b = log.pop()
            (chosen if op == 'c' else banned)[a].discard(b)

    def propagate(log):
        """полный ресcan до неподвижной точки, изменения — через журнал откатов"""
        changed = True
        while changed:
            changed = False
            for rc in keys:
                need = req[rc] - len(chosen[rc])
                if need < 0:
                    return False
                avail = [x for x in neigh[rc] if x not in chosen[rc] and x not in banned[rc]]
                if need == 0:
                    for x in avail:
                        if x not in banned[rc]:
                            banned[rc].add(x); banned[x].add(rc)
                            log.append(('b', rc, x)); log.append(('b', x, rc))
                            changed = True
                elif len(avail) < need:
                    return False
                elif len(avail) == need:
                    for x in avail:
                        if x not in chosen[rc]:
                            chosen[rc].add(x); chosen[x].add(rc)
                            log.append(('c', rc, x)); log.append(('c', x, rc))
                            changed = True
            for rc in keys:
                if req[rc] != 1 or len(chosen[rc]) != 1:
                    continue
                chain = [rc]
                cur = next(iter(chosen[rc]))
                prev = rc
                while True:
                    if req[cur] == 1:
                        a, b = cells[rc], cells[cur]
                        if a['color'] != b['color'] or a['clue'] != b['clue'] or len(chain) + 1 != a['clue']:
                            return False
                        added = False
                        for x in chain + [cur]:
                            for y in neigh[x]:
                                if y not in chosen[x] and y not in banned[x]:
                                    banned[x].add(y); banned[y].add(x)
                                    log.append(('b', x, y)); log.append(('b', y, x))
                                    added = True
                        if added:
                            changed = True
                        break
                    if len(chosen[cur]) != 2:
                        break
                    nxt = [x for x in chosen[cur] if x != prev][0]
                    if nxt in chain:
                        return False
                    chain.append(nxt)
                    prev, cur = cur, nxt
                    if len(chain) > rows * cols:
                        return False
        return True

    def connected(a, b):
        stack = [a]; seen = {a}
        while stack:
            x = stack.pop()
            if x == b:
                return True
            stack.extend(y for y in chosen[x] if y not in seen)
            seen.update(chosen[x])
        return False

    def search():
        nonlocal sol_count
        if sol_count >= limit or aborted[0]:
            return
        nodes[0] += 1
        if nodes[0] > node_budget or (deadline is not None and _t.time() > deadline):
            aborted[0] = True
            if progress:
                print(f'  солвер: бюджет исчерпан ({nodes[0]} узлов, '
                      f'{_t.time()-t0:.1f} с)', flush=True)
            return
        if progress and nodes[0] % 50000 == 0:
            print(f'  солвер: {nodes[0]} узлов, решений {sol_count}, '
                  f'{_t.time()-t0:.1f} с...', flush=True)
        log = []
        base = len(log)
        if not propagate(log):
            undo(log, base)
            return
        best = None
        for rc in keys:
            need = req[rc] - len(chosen[rc])
            if need > 0:
                avail = [x for x in neigh[rc] if x not in chosen[rc] and x not in banned[rc]]
                if not avail:
                    undo(log, base)
                    return
                if best is None or len(avail) < len(best[1]):
                    best = (rc, avail)
        if best is None:
            if full_check(chosen):
                sol_count += 1
                sols.append({frozenset((rc, x)) for rc in keys for x in chosen[rc]})
            undo(log, base)
            return
        rc, avail = best
        if prefer is not None:
            avail.sort(key=lambda x: 0 if frozenset((rc, x)) in prefer else 1)
        for x in avail:
            if connected(rc, x):
                continue
            chosen[rc].add(x); chosen[x].add(rc)
            log.append(('c', rc, x)); log.append(('c', x, rc))
            search()
            undo(log, base)
            if sol_count >= limit or aborted[0]:
                undo(log, base)
                return
        undo(log, base)

    chosen = {rc: set() for rc in keys}
    banned = {rc: set() for rc in keys}
    if forced:
        for a, b in forced:
            chosen[a].add(b); chosen[b].add(a)
    for (r, c), cl in cells.items():
        for bit, (dr, dc) in WBIT.items():
            nb = (r + dr, c + dc)
            if cl['wall'] & bit and nb in banned:
                banned[(r, c)].add(nb); banned[nb].add((r, c))
    search()
    if aborted[0]:
        return sol_count, [], True
    return sol_count, sols, False

def intended_pairs(cells):
    pairs = []
    for (r, c), cl in cells.items():
        for code in (cl['hi'], cl['lo']):
            if code:
                dr, dc = CODE_DIR[code]
                pairs.append(((r, c), (r + dr, c + dc)))
    return pairs

def repair_walls(grid, cells, rows, cols, max_repairs=200, deadline=None, progress=False,
                   solver=None):
    """Ищет решения, отличные от канонического; на ребро альтернативы
    (не из канона) ставит стену. Возвращает число добавленных стен."""
    intended = intended_pairs(cells)
    intended_edges = {frozenset(e) for e in intended}
    repairs = 0
    while repairs < max_repairs:
        cnt, sols, ab = solver(cells, rows, cols, limit=2,
                               deadline=deadline, progress=progress,
                               prefer=intended_edges, node_budget=30000)
        if ab or cnt <= 1:
            break
        alt = next((s for s in sols if s != intended_edges), None)
        if alt is None:
            break
        diff = [e for e in alt if e not in intended_edges]
        if not diff:
            break
        (a, b) = tuple(diff[0])
        ar, ac = a; br, bc = b
        cells[a]['wall'] |= WALL_BIT[DIR_CODE[(br - ar, bc - ac)]]
        cells[b]['wall'] |= WALL_BIT[DIR_CODE[(ar - br, ac - bc)]]
        repairs += 1
    return repairs

def prewall_parallel(grid, paths, cells, min_color_share=0.15, density=0.15, seed=0):
    # Разреженные стены на параллельных контактах крупных цветов.
    # У оригиналов застенено ~15% контактов больших областей (114.ppz) —
    # берём это как плотность; солвер/ремонт добивает остаток.
    import random
    rng = random.Random(seed)
    rows, cols = len(grid), len(grid[0])
    color_cnt = Counter(x for row in grid for x in row)
    big = {k for k, v in color_cnt.items() if v >= min_color_share * rows * cols}
    owner = {rc: pi for pi, p in enumerate(paths) for rc in p}
    n = 0
    for (r, c), cl in cells.items():
        if cl['color'] not in big:
            continue
        for dr, dc in ((0, 1), (1, 0)):
            nb = (r + dr, c + dc)
            if nb not in cells:
                continue
            oa, ob = owner.get((r, c)), owner.get(nb)
            if oa is None or ob is None or oa == ob:
                continue
            if cl['color'] != cells[nb]['color']:
                continue
            para = False
            for pdr, pdc in ((dc, dr), (-dc, -dr)):
                if owner.get((r + pdr, c + pdc)) == oa and owner.get((nb[0] + pdr, nb[1] + pdc)) == ob:
                    para = True
                    break
            if para and rng.random() < density:
                bit = WALL_BIT[DIR_CODE[(dr, dc)]]
                if not (cl['wall'] & bit):
                    cl['wall'] |= bit
                    cells[nb]['wall'] |= WALL_BIT[DIR_CODE[(-dr, -dc)]]
                    n += 1
    return n


# ---------- 3. сборка записей ----------

DIR_CODE = {(0, -1): 1, (-1, 0): 2, (0, 1): 3, (1, 0): 4}   # W,N,E,S
CODE_DIR = {v: k for k, v in DIR_CODE.items()}
WALL_BIT = {1: 1, 2: 2, 3: 4, 4: 8}   # по коду направления -> бит стены

def build_cells(grid, paths, squares):
    """-> cells {(r,c): dict(color, clue, hi, lo, wall)}"""
    rows, cols = len(grid), len(grid[0])
    owner = {}
    for pi, path in enumerate(paths):
        for r, c in path:
            owner[(r, c)] = pi
    cells = {}
    for pi, path in enumerate(paths):
        n = len(path)
        for i, (r, c) in enumerate(path):
            prev = path[i - 1] if i > 0 else None
            nxt = path[i + 1] if i < n - 1 else None
            hi = DIR_CODE[(nxt[0] - r, nxt[1] - c)] if nxt else 0   # hi = next
            lo = DIR_CODE[(prev[0] - r, prev[1] - c)] if prev else 0  # lo -> prev
            cells[(r, c)] = dict(color=grid[r][c], clue=n if i in (0, n - 1) else 0,
                                 hi=hi, lo=lo, wall=0)
    for (r, c) in squares:
        cells[(r, c)] = dict(color=grid[r][c], clue=1, hi=0, lo=0, wall=0)
    # стены не ставим: единственность добивается ремонтом (repair_walls)
    return cells

# ---------- 4. запись .ppz ----------

def write_ppz(fn, rows, cols, palette, cells):
    body = struct.pack('>HHB', rows, cols, len(palette))
    body += b'\x00' * 15
    body += b'R'
    body += b'\x00' * 4
    body += b'\x00' * 16
    for a, r, g, b in palette:
        body += struct.pack('>I', (a << 24) | (r << 16) | (g << 8) | b)
    for r in range(rows):
        for c in range(cols):
            cl = cells[(r, c)]
            body += bytes([(cl['color'] << 3) | cl['hi'],
                           (cl['clue'] << 3) | cl['lo'],
                           cl['wall']])
    head = b'PPZ ' + bytes([2, 1]) + struct.pack('>I', len(body))
    with open(fn, 'wb') as f:
        f.write(head + body)
    return len(head + body)

# ---------- 5. валидатор (зеркало правил игры) ----------

def validate(grid, paths, squares, cells):
    rows, cols = len(grid), len(grid[0])
    errs = []
    adj = {}
    for (r, c), cl in cells.items():
        ns = []
        for code in (cl['hi'], cl['lo']):
            if code:
                dr, dc = CODE_DIR[code]
                if 0 <= r + dr < rows and 0 <= c + dc < cols:
                    ns.append((r + dr, c + dc))
                else:
                    errs.append(f'стрелка за границей ({r},{c})')
        adj[(r, c)] = ns
    for (r, c), cl in cells.items():
        d = len(adj[(r, c)])
        if cl['clue'] == 1:
            if d: errs.append(f'квадратик со стрелками ({r},{c})')
        elif cl['clue'] > 1:
            if d != 1: errs.append(f'подсказка ({r},{c}) степень {d}')
        elif d != 2:
            errs.append(f'клетка ({r},{c}) степень {d}')
        for (nr, nc) in adj[(r, c)]:
            s = (nr - r, nc - c)
            if cl['wall'] & WALL_BIT[DIR_CODE[s]]:
                errs.append(f'путь через стену ({r},{c})')
    # обход путей
    visited, total = set(), 0
    for pi, path in enumerate(paths):
        a, b = path[0], path[-1]
        for end in (a, b):
            if cells[end]['clue'] != len(path):
                errs.append(f'путь {pi}: подсказка {cells[end]["clue"]} != длине {len(path)}')
        cur, prev, ln = a, None, 1
        visited.add(cur)
        while cur != b:
            nx = [x for x in adj[cur] if x != prev]
            if not nx: errs.append(f'тупик пути {pi}'); break
            nxt = nx[0]
            if nxt in visited: errs.append(f'петля пути {pi}'); break
            visited.add(nxt); prev, cur = cur, nxt; ln += 1
        if ln != len(path): errs.append(f'путь {pi}: обошёл {ln} != {len(path)}')
        total += 1
    if len(visited) + len(squares) != rows * cols:
        errs.append(f'покрытие {len(visited)}+{len(squares)} != {rows*cols}')
    ssum = sum(2 * len(p) for p in paths)
    if ssum != 2 * (rows * cols - len(squares)):
        errs.append(f'сумма {ssum} != 2×(клетки−квадратики)')
    return errs

# ---------- 6. превью ----------

def preview(grid, palette, cells, fn, cell=26):
    rows, cols = len(grid), len(grid[0])
    W, H = cols * cell + 20, rows * cell + 20
    img = Image.new('RGB', (W, H), (52, 52, 156))
    dr = ImageDraw.Draw(img)
    dr.rectangle([10, 10, W - 10, H - 10], fill=(205, 203, 228))
    for i in range(cols + 1):
        dr.line([10 + i * cell, 10, 10 + i * cell, H - 10], fill=(180, 179, 210))
    for j in range(rows + 1):
        dr.line([10, 10 + j * cell, W - 10, 10 + j * cell], fill=(180, 179, 210))
    rgb = [(r, g, b) for a, r, g, b in palette]
    for (r, c), cl in cells.items():
        x, y = 10 + c * cell, 10 + r * cell
        if cl['wall']:
            dr.line([x, y + cell * .25, x, y + cell * .75], fill=(216, 30, 30), width=max(2, cell // 8)) if cl['wall'] & 1 else None
            dr.line([x + cell * .25, y, x + cell * .75, y], fill=(216, 30, 30), width=max(2, cell // 8)) if cl['wall'] & 2 else None
            dr.line([x + cell, y + cell * .25, x + cell, y + cell * .75], fill=(216, 30, 30), width=max(2, cell // 8)) if cl['wall'] & 4 else None
            dr.line([x + cell * .25, y + cell, x + cell * .75, y + cell], fill=(216, 30, 30), width=max(2, cell // 8)) if cl['wall'] & 8 else None
    for (r, c), cl in cells.items():
        cx, cy = 10 + c * cell + cell / 2, 10 + r * cell + cell / 2
        if cl['clue'] == 1:
            s = cell * .55
            dr.rectangle([cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2], fill=rgb[cl['color']])
        elif cl['clue'] > 1:
            R = cell * .38
            dr.ellipse([cx - R, cy - R, cx + R, cy + R], fill=rgb[cl['color']])
            rr, gg, bb = rgb[cl['color']]
            dark = (0.3 * rr + 0.59 * gg + 0.11 * bb) / 255 < 0.62
            dr.text((cx, cy), str(cl['clue']), fill=(255, 255, 255) if dark else (26, 26, 26),
                    anchor='mm', font_size=int(cell * .45))
    img.save(fn)

# ---------- 7. конвейер ----------

def auto_params(img, max_size, n_colors, target, bendiness):
    """Значения по умолчанию с учётом ограничений игры:
    цветов 2..30 (5 бит на цвет, слот 0 резерв), путь 2..31 (5 бит подсказки),
    bendiness 0..1. Длина ~5 как в реальных уровнях; для больших полей чуть длиннее
    (меньше путей -> меньше контактов и быстрее солвер)."""
    w, h = img.size
    s = min(1.0, max_size / max(w, h))
    area = round(w * s) * round(h * s)
    if n_colors is None:
        n_colors = 10
    if target is None:
        target = 5 if area <= 2500 else (6 if area <= 8000 else 7)
    if bendiness is None:
        bendiness = 0.85
    return (max(2, min(30, n_colors)), max(2, min(31, target)),
            max(0.0, min(1.0, bendiness)))

def wall_all_contacts(cells, owner):
    """Стены на ВСЕХ рёбрах между путями одного цвета —
    единственность по построению (тяжёлый, но гарантированный вариант)."""
    n = 0
    for (r, c), cl in cells.items():
        for dr, dc in ((0, 1), (1, 0)):
            nb = (r + dr, c + dc)
            if nb not in cells:
                continue
            oa, ob = owner.get((r, c)), owner.get(nb)
            if oa is None or ob is None or oa == ob:
                continue
            if cl['color'] != cells[nb]['color']:
                continue
            cl['wall'] |= WALL_BIT[DIR_CODE[(dr, dc)]]
            cells[nb]['wall'] |= WALL_BIT[DIR_CODE[(-dr, -dc)]]
            n += 1
    return n

def pick_solver(name='auto'):
    """-> callable(cells, rows, cols, limit=..., deadline=..., progress=...,
    prefer=..., node_budget=...) с возвратом (count, sols, aborted)."""
    if name in ('auto', 'sat'):
        try:
            import pysat.solvers  # noqa: F401 — сама проверка зависимости
            import ppz_sat
            return ppz_sat.count_solutions
        except ImportError:
            if name == 'sat':
                raise RuntimeError('python-sat не установлен: pip install python-sat')
    return count_solutions

def generate(img, out_ppz, max_size=150, n_colors=None, preview_fn=None,
             target=None, seed=0, check=True, bendiness=None,
             check_time=15.0, safe_walls=False, progress=True, solver='auto',
             export_img=None, try_seeds=1, merge_thresh=24, prewall_density=None):
    import time as _t
    def say(msg):
        if progress:
            print(msg, flush=True)
    n_colors, target, bendiness = auto_params(img, max_size, n_colors, target, bendiness)
    say(f'[1/5] изображение → сетка ({n_colors} цветов)...')
    grid, palette = image_to_grid(img, max_size, n_colors, merge_thresh=merge_thresh)
    rows, cols = len(grid), len(grid[0])
    if export_img:
        prgb = [(r, g, b) for a, r, g, b in palette]
        base = export_img.rsplit('.png', 1)[0]
        # 4× — как в файлах игры
        pim = Image.new('RGB', (cols * 4, rows * 4))
        for r in range(rows):
            for c in range(cols):
                pim.paste(prgb[grid[r][c]], (c * 4, r * 4, (c + 1) * 4, (r + 1) * 4))
        pim.save(base + '_4x.png')
        # 1× — «сырые» пиксели уровня, клетка = 1 px
        fim = Image.new('RGB', (cols, rows))
        for r in range(rows):
            for c in range(cols):
                fim.putpixel((c, r), prgb[grid[r][c]])
        fim.save(base + '_1x.png')
        say(f'      картинки уровня → {base}_4x.png, {base}_1x.png')
    repairs = 0
    unique = None
    prewalled = 0
    if safe_walls:
        say(f'[2/5] разбиение на пути (цель {target}, seed {seed})...')
        paths, squares = decompose(grid, target=target, seed=seed, bendiness=bendiness)
        say(f'      {len(paths)} путей, {len(squares)} квадратиков')
        cells = build_cells(grid, paths, squares)
        errs = validate(grid, paths, squares, cells)
        if errs:
            raise RuntimeError('валидация не пройдена: ' + '; '.join(errs[:5]))
        owner = {rc: pi for pi, p in enumerate(paths) for rc in p}
        wn = wall_all_contacts(cells, owner)
        say(f'[3/5] стены на всех контактах одноцветных путей: {wn} (единственность по построению)')
        unique = True
    elif check:
        solver_fn = pick_solver(solver)
        is_backtrack = solver_fn is count_solutions
        can_check = rows * cols <= 1500 or not is_backtrack
        tries = max(1, try_seeds) if can_check else 1
        if can_check and rows * cols > 3000 and tries > 10:
            say(f'      замечание: поле {rows*cols} клеток × {tries} сидов — '
                f'на попытку останется мало времени; лучше --try-seeds 5..15')
        say(f'[2/5] разбиение на пути (цель {target}, сидов: {tries})...')
        best = None
        accepted = False
        search_budget = check_time / 2
        t_search = _t.time()
        for t in range(tries):
            s = seed + t
            # адаптивный порог: остаток бюджета / оставшиеся попытки (минимум 5 с)
            per_try = max(5.0, (search_budget - (_t.time() - t_search)) / (tries - t))
            paths, squares = decompose(grid, target=target, seed=s, bendiness=bendiness)
            cells = build_cells(grid, paths, squares)   # БЕЗ стен: ищем сид,
            errs = validate(grid, paths, squares, cells)  # однозначный сам по себе
            if errs:
                raise RuntimeError('валидация не пройдена: ' + '; '.join(errs[:5]))
            if best is None:
                best = (s, paths, squares, cells)
            if not can_check:
                break
            cnt, _, ab = solver_fn(cells, rows, cols, limit=2,
                                   deadline=_t.time() + per_try,
                                   prefer={frozenset(e) for e in intended_pairs(cells)},
                                   node_budget=30000)
            ok = (not ab) and cnt == 1
            if progress:
                say(f'      сид {s}: ' + ('единственное решение без стен' if ok else
                    (f'не подтверждено (лимит {per_try:.0f} с)' if ab else f'{cnt} решения — нужны стены')))
            if ok:
                best = (s, paths, squares, cells)
                accepted = True
                unique = True
                repairs = 0
                break
        seed, paths, squares, cells = best
        if can_check and not accepted:
            dens = prewall_density if prewall_density is not None else \
                (0.15 if not is_backtrack else 1.0)
            prewalled = prewall_parallel(grid, paths, cells, seed=seed, density=dens)
            say(f'      ни один из {tries} сидов не уникален без стен; лучший — сид {seed}: '
                f'пре-воллинг {prewalled} стен + ремонт')
            say(f'[3/5] ремонт лучшего сида (лимит {check_time:.0f} с)...')
            deadline = _t.time() + check_time
            repairs = repair_walls(grid, cells, rows, cols,
                                   deadline=_t.time() + check_time / 2,
                                   progress=progress, solver=solver_fn)
            cnt, _, ab = solver_fn(cells, rows, cols, limit=2,
                                   deadline=deadline, progress=progress,
                                   prefer={frozenset(e) for e in intended_pairs(cells)},
                                   node_budget=30000)
            unique = None if ab else (cnt == 1)
            if not unique:
                say('      совет: для гарантии используйте --safe-walls '
                    '(единственность по построению) или больше --try-seeds')
            say(f'      ремонт: {repairs} стен; ' +
                ({True: 'единственное решение', False: 'НЕ ЕДИНСТВЕННО',
                  None: 'не подтверждено (лимит времени)'}[unique]))
        elif not can_check:
            say('[3/5] солвер пропущен: большое поле (>1500 клеток) и бэктрек; '
                'установите python-sat или используйте --safe-walls')
            unique = None
    else:
        say(f'[2/5] разбиение на пути (цель {target}, seed {seed})...')
        paths, squares = decompose(grid, target=target, seed=seed, bendiness=bendiness)
        say(f'      {len(paths)} путей, {len(squares)} квадратиков')
        cells = build_cells(grid, paths, squares)
        errs = validate(grid, paths, squares, cells)
        if errs:
            raise RuntimeError('валидация не пройдена: ' + '; '.join(errs[:5]))
        say('[3/5] проверка отключена (--no-check)')

    say(f'[4/5] запись {out_ppz}...')
    n = write_ppz(out_ppz, rows, cols, palette, cells)
    if preview_fn:
        preview(grid, palette, cells, preview_fn)
    say('[5/5] независимая проверка файла...')
    verr, _ = verify_ppz(out_ppz)
    if verr:
        raise RuntimeError('файл не прошёл проверку: ' + '; '.join(verr[:5]))
    say('      файл корректен.')
    import statistics
    bl = []
    for p in paths:
        b = sum(1 for k in range(1, len(p) - 1)
                if (p[k][0] - p[k-1][0], p[k][1] - p[k-1][1]) != (p[k+1][0] - p[k][0], p[k+1][1] - p[k][1]))
        bl.append(b)
    return dict(rows=rows, cols=cols, colors=len(palette) - 1,
                paths=len(paths), squares=len(squares),
                walls=sum(bin(cl['wall']).count('1') for cl in cells.values()) // 2,
                repairs=repairs, prewalled=prewalled, unique=unique, target=target,
                avg_len=round(sum(len(p) for p in paths) / max(1, len(paths)), 2),
                avg_bends=round(statistics.mean(bl), 2) if bl else 0,
                bytes=n)

def main():
    ap = argparse.ArgumentParser(description='Генератор уровней PathPix (.ppz)')
    ap.add_argument('input'); ap.add_argument('output')
    ap.add_argument('--max', type=int, default=150)
    ap.add_argument('--colors', type=int, default=None, help='по умолчанию авто (10)')
    ap.add_argument('--prewall-density', type=float, default=None,
                    help='доля параллельных контактов под пре-стенами (0..1); '
                         'умолчание 0.15 с SAT, 1.0 с бэктреком')
    ap.add_argument('--merge-thresh', type=int, default=24,
                    help='слияние визуально одинаковых цветов палитры (0 = отключить; '
                         'для чистых PNG 16-24, зашумленных JPEG 32-40)')
    ap.add_argument('--target', type=int, default=None, help='желаемая длина пути, авто: 5-7')
    ap.add_argument('--seed', default=0, help='целое или "random"')
    ap.add_argument('--bendiness', type=float, default=None, help='склонность к поворотам, авто 0.7')
    ap.add_argument('--no-check', action='store_true', help='не проверять единственность солвером')
    ap.add_argument('--time', type=float, default=15.0, help='лимит времени солвера, сек')
    ap.add_argument('--safe-walls', action='store_true',
                    help='единственность по построению: стены на всех контактах одноцветных путей')
    ap.add_argument('--quiet', action='store_true', help='без вывода этапов')
    ap.add_argument('--try-seeds', type=int, default=1,
                    help='сколько сидов перебрать в поиске однозначности без ремонтных стен')
    ap.add_argument('--solver', choices=['auto', 'sat', 'backtrack'], default='auto',
                    help='auto: SAT, если установлен python-sat, иначе бэктрек')
    ap.add_argument('--preview')
    ap.add_argument('--export-img', metavar='BASE',
                    help='картинка уровня (квантованная): BASE_4x.png (4× от сетки, как в игре) '
                         'и BASE_1x.png (1× — пиксели уровня без масштаба)')
    a = ap.parse_args()
    seed = a.seed
    if isinstance(seed, str) and seed.lower() == 'random':
        import random as _rnd
        seed = _rnd.SystemRandom().randrange(2 ** 31)
        print(f'сид: {seed}', flush=True)
    img = Image.open(a.input)
    st = generate(img, a.output, a.max, a.colors, a.preview,
                  target=a.target, seed=seed, check=not a.no_check,
                  bendiness=a.bendiness, check_time=a.time,
                  safe_walls=a.safe_walls, progress=not a.quiet, solver=a.solver,
                  export_img=a.export_img, try_seeds=a.try_seeds,
                  merge_thresh=a.merge_thresh, prewall_density=a.prewall_density)
    u = {True: 'единственное решение', False: 'НЕ ЕДИНСТВЕННО', None: 'не проверялось (большое поле)'}[st['unique']]
    print(f"OK: {st['rows']}×{st['cols']}, цветов {st['colors']}, путей {st['paths']}, "
          f"квадратиков {st['squares']}, стен {st['walls']} (ремонт: {st['repairs']}), "
          f"цель {st['target']}, ср. длина {st['avg_len']}, ср. поворотов {st['avg_bends']}, "
          f"{u}, файл {st['bytes']} байт")



# ---------- 8. независимая проверка записанного .ppz ----------

def verify_ppz(fn):
    """Парсит файл с нуля и прогоняет полную симуляцию правил игры."""
    data = open(fn, 'rb').read()
    assert data[:4] == b'PPZ ', 'маркер'
    cv, bv = data[4], data[5]
    hs = {2: 10, 3: 18, 4: 8}[cv]
    bs = struct.unpack('>I', data[6:10])[0] if cv in (2, 3) else struct.unpack('>H', data[6:8])[0]
    assert len(data) - hs == bs, 'размер тела'
    assert bv == 1, 'тело v1'
    o = hs
    rows, cols, nc = struct.unpack('>HHB', data[o:o + 5]); o += 5 + 15
    geom = chr(data[o]); o += 1 + 4 + 16
    pal = [struct.unpack('>I', data[o + 4 * i:o + 4 * i + 4])[0] for i in range(nc)]; o += 4 * nc
    assert len(data) - o == rows * cols * 3, 'хвост'
    cells = {}
    for r in range(rows):
        for c in range(cols):
            b0, b1, b2 = data[o], data[o + 1], data[o + 2]; o += 3
            cells[(r, c)] = dict(color=b0 >> 3, hi=b0 & 7, clue=b1 >> 3, lo=b1 & 7, wall=b2)
    errs = []
    adj = {}
    for (r, c), cl in cells.items():
        ns = []
        for code in (cl['hi'], cl['lo']):
            if code:
                if code > 4: errs.append(f'код стрелки {code} ({r},{c})'); continue
                dr, dc = CODE_DIR[code]
                if not (0 <= r + dr < rows and 0 <= c + dc < cols):
                    errs.append(f'стрелка за границей ({r},{c})')
                else:
                    ns.append((r + dr, c + dc))
        adj[(r, c)] = ns
    squares = 0
    for (r, c), cl in cells.items():
        d = len(adj[(r, c)])
        if cl['clue'] == 1:
            squares += 1
            if d: errs.append(f'квадратик со стрелками ({r},{c})')
        elif cl['clue'] > 1:
            if d != 1: errs.append(f'подсказка ({r},{c}) степень {d}')
        elif d != 2:
            errs.append(f'клетка ({r},{c}) степень {d}')
        if cl['color'] >= nc: errs.append(f'цвет вне палитры ({r},{c})')
        for (nr, ncc) in adj[(r, c)]:
            s = DIR_CODE[(nr - r, ncc - c)]
            if cl['wall'] & WALL_BIT[s]: errs.append(f'путь через стену ({r},{c})')
        for code, bit in WALL_BIT.items():
            if cl['wall'] & bit:
                dr, dc = CODE_DIR[code]
                nr, ncc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= ncc < cols):
                    errs.append(f'стена на краю ({r},{c})')
                elif not (cells[(nr, ncc)]['wall'] & WALL_BIT[{1: 3, 3: 1, 2: 4, 4: 2}[code]]):
                    errs.append(f'стена без пары ({r},{c})')
    visited, paths_n, sumclues = set(), 0, 0
    for (r, c), cl in cells.items():
        if cl['clue'] > 1 and (r, c) not in visited:
            paths_n += 1
            cur, prev, ln = (r, c), None, 1
            visited.add(cur); sumclues += 2 * cl['clue']
            while True:
                nx = [x for x in adj[cur] if x != prev]
                if not nx: errs.append(f'тупик от ({r},{c})'); break
                nxt = nx[0]
                if nxt in visited: errs.append(f'петля от ({r},{c})'); break
                visited.add(nxt); prev, cur = cur, nxt; ln += 1
                if cells[cur]['clue'] > 1: break
            if cells[cur]['clue'] > 1:
                if cells[cur]['clue'] != cells[(r, c)]['clue'] or ln != cells[(r, c)]['clue']:
                    errs.append(f'длина пути от ({r},{c}): {ln} != {cells[(r,c)]["clue"]}')
                if cells[cur]['color'] != cells[(r, c)]['color']:
                    errs.append(f'цвета концов разные ({r},{c})')
    for (r, c), cl in cells.items():
        if cl['clue'] != 1 and (r, c) not in visited:
            errs.append(f'клетка ({r},{c}) вне путей (clue={cl["clue"]})')
    if sumclues != 2 * (rows * cols - squares):
        errs.append(f'сумма подсказок {sumclues} != 2×{rows*cols-squares}')
    return errs, dict(rows=rows, cols=cols, colors=nc - 1, paths=paths_n,
                      squares=squares, geom=geom)


if __name__ == '__main__':
    main()
