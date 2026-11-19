import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import layout
import pacman
import search
from tspAgents import TSPProblem, tspHeuristic
from game import Actions


def draw_map(name):
    game = pacman.GameState()
    game.initialize(layout.getLayout(name), 0)
    walls = game.getWalls()
    goals = [(1, 1), (1, walls.height-2), (walls.width-2, 1), (walls.width-2, walls.height-2)]
    problem = TSPProblem(game, goals)
    path = search.aStarSearch(problem, tspHeuristic)
    points = [problem.start]
    for a in path:
        dx,dy = Actions.directionToVector(a)
        x,y = points[-1]
        points.append((int(x+dx), int(y+dy)))

    fig, ax = plt.subplots(figsize=(8.1, 4.8) if name == 'mediumCorners' else (7.5, 5.3), dpi=180)
    ax.set_facecolor('#f6f9fd')
    for x in range(walls.width):
        for y in range(walls.height):
            if walls[x][y]:
                ax.add_patch(Rectangle((x-.5, y-.5),1,1,facecolor='#23354d',edgecolor='#23354d'))
    xs,ys = zip(*points)
    ax.plot(xs,ys,color='#159a79',linewidth=2.6,solid_capstyle='round',label='TSP A* path',zorder=5)
    for i,pt in enumerate(goals,1):
        ax.scatter(*pt,c='#e3aa31',edgecolors='#26364a',s=95,zorder=7)
        ax.annotate(str(i), (pt[0]+.25,pt[1]+.3), color='#26364a',fontsize=8,fontweight='bold',zorder=8)
    ax.scatter(*problem.start,marker='*',s=240,c='#de634b',edgecolors='#fff',linewidths=.7,zorder=9,label='Start')
    ax.set_xlim(-.5, walls.width-.5); ax.set_ylim(-.5, walls.height-.5)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'{name} — TSP A*: {len(path)} moves, {problem._expanded} expanded states',fontsize=11,fontweight='semibold')
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.tight_layout()
    fn=f'outputs/figures/{name}_TSP.png'
    fig.savefig(fn,dpi=180,facecolor='white',bbox_inches='tight')
    plt.close(fig)
    print(fn)

for board in ('tinyCorners','mediumCorners'):
    draw_map(board)
