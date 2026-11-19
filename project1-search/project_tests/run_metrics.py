import csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import layout
import pacman
import search
from searchAgents import PositionSearchProblem, manhattanHeuristic
from tspAgents import TSPProblem, tspHeuristic


def get_game(board):
    state = pacman.GameState()
    state.initialize(layout.getLayout(board), 0)
    return state


rows = []
for board in ('mediumMaze', 'bigMaze'):
    game = get_game(board)
    methods = (
        ('DFS', search.depthFirstSearch),
        ('BFS', search.breadthFirstSearch),
        ('UCS', search.uniformCostSearch),
        ('A* (Manhattan)', lambda p: search.aStarSearch(p, manhattanHeuristic)),
    )
    for name, fn in methods:
        problem = PositionSearchProblem(game, warn=False, visualize=False)
        path = fn(problem)
        rows.append(['Navigation', board, name, 'open', len(path), problem.getCostOfActions(path), problem._expanded])

for board in ('tinyCorners', 'mediumCorners'):
    game = get_game(board)
    walls = game.getWalls()
    corners = [(1, 1), (1, walls.height - 2), (walls.width - 2, 1), (walls.width - 2, walls.height - 2)]
    for mode in (False, True):
        methods = (
            ('BFS', search.breadthFirstSearch),
            ('UCS', search.uniformCostSearch),
            ('A* (MST + maze distance)', lambda p: search.aStarSearch(p, tspHeuristic)),
        )
        for name, fn in methods:
            problem = TSPProblem(game, corners, mode)
            path = fn(problem)
            rows.append(['TSP', board, name, 'closed' if mode else 'open',
                         len(path), problem.getCostOfActions(path), problem._expanded])

with open('outputs/KET_QUA_THUC_NGHIEM.csv', 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f)
    w.writerow(['Bai_toan', 'Ban_do', 'Thuat_toan', 'Che_do', 'So_buoc', 'Tong_chi_phi', 'Trang_thai_mo_rong'])
    w.writerows(rows)
for row in rows:
    print(' | '.join(map(str, row)))
