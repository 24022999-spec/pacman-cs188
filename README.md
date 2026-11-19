# Pacman AI – CS188 Project 1 (Search) & Project 2 (Multi-Agent Search)

Bài làm hai project Pacman của môn CS188 (UC Berkeley, Fall 2025):

| Thư mục | Nội dung | File nhóm viết | Autograder |
|---|---|---|---|
| `project1-search/` | DFS, BFS, UCS, A*, Corners Problem, Food Search, Closest Dot (Q1–Q8) | `search.py`, `searchAgents.py` | **26/25** (có điểm thưởng Q7) |
| `project2-multiagent/` | Reflex Agent, Minimax, Alpha-Beta, Expectimax, hàm đánh giá (Q1–Q5) | `multiAgents.py` | **25/25** |

Mỗi project là một bộ mã nguồn riêng; các file khác (lõi trò chơi, bộ chấm, `test_cases/`) giữ nguyên như bản gốc.

## Yêu cầu

Python 3 (đã chạy thử với Python 3.11 và 3.13). Không cần cài thêm thư viện.

## Chạy Autograder

```bash
cd project1-search
python autograder.py --no-graphics     # toàn bộ Q1–Q8
python autograder.py -q q6             # một câu

cd ../project2-multiagent
python autograder.py --no-graphics     # toàn bộ Q1–Q5
```

Log của lần chạy gần nhất: `project1-search/P1_Autograder.txt`, `project2-multiagent/P2_Autograder.txt`.

## Một số lệnh chạy thử

```bash
# Project 1
cd project1-search
python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs
python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
python pacman.py -l mediumCorners -p AStarCornersAgent -z 0.5
python pacman.py -l trickySearch -p AStarFoodSearchAgent
python pacman.py -l bigSearch -p ClosestDotSearchAgent -z .5

# Project 2
cd ../project2-multiagent
python pacman.py -p ReflexAgent -l testClassic
python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=4
python pacman.py -p ExpectimaxAgent -l trappedClassic -a depth=3 -q -n 10
```

## Kết quả chính

**Project 1**

| Bài toán | Thuật toán | Số bước | Trạng thái mở rộng |
|---|---|---:|---:|
| mediumMaze | DFS / BFS / UCS / A* (Manhattan) | 130 / 68 / 68 / 68 | 146 / 269 / 269 / 221 |
| mediumCorners | A* + `cornersHeuristic` | 106 | 189 |
| trickySearch | A* + `foodHeuristic` | 60 | 205 |
| bigSearch | ClosestDotSearchAgent | 350 | – |

**Project 2**

| Câu | Điểm | Ván Autograder |
|---|---|---|
| Q1 Reflex | 4/4 | thắng 10/10, TB 1.282,7 |
| Q2 Minimax / Q3 Alpha-Beta / Q4 Expectimax | 5/5 mỗi câu | test kiểm tra từng nước đi và số trạng thái |
| Q5 Evaluation Function | 6/6 | thắng 10/10, TB 1.151,4 |

## Phần bổ sung (ngoài phạm vi chấm điểm)

- `project1-search/tspAgents.py`: lộ trình đi qua nhiều điểm (TSP), có chế độ quay về điểm xuất phát.
  Ví dụ: `python pacman.py -l mediumCorners -p TSPAgent -a mode=closed`. Test: `python -m unittest discover -s project_tests -v`.
- `project2-multiagent/project_tests/`: 12 unit test nội bộ (`python -m unittest discover -s project_tests -p "test_*.py" -v`).

## Ghi công

Các project Pacman AI được phát triển tại UC Berkeley (http://ai.berkeley.edu). The core projects and autograders were primarily created by John DeNero and Dan Klein; student side autograding was added by Brad Miller, Nick Hay, and Pieter Abbeel. Thông báo license gốc được giữ nguyên ở đầu mỗi file.
