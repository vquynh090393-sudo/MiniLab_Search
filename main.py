# Import libraries
from collections import deque   # BFS frontier (queue)
import matplotlib.pyplot as plt
import networkx as nx           # tạo và thao tác đồ thị
import heapq                    # priority queue cho A*
import math                     # sqrt cho heuristic

# 2. Create a small graph symbolizing Vietnam
# 2.1 G là đồ thị vô hướng (Graph).Edges định nghĩa kết nối tuyến tính HaNoi—Hue—DaNang—SaiGon—CanTho (cộng thêm HaNoi—HaiPhong).
cities = ["HaNoi", "Hue", "DaNang", "SaiGon", "CanTho", "HaiPhong"]

edges = [
    ("HaNoi", "HaiPhong"),
    ("HaNoi", "Hue"),
    ("Hue", "DaNang"),
    ("DaNang", "SaiGon"),
    ("SaiGon", "CanTho")
]

# 2.2 Build the graph
G = nx.Graph()
G.add_nodes_from(cities)
G.add_edges_from(edges)

# positions for heuristic (rough map)
# Toạ độ pos dùng để tính heuristic (khoảng cách Euclid):pos ở đây chỉ ước lượng — dùng để tính h(n).
pos = {
    "DaNang": (0, 0),      # Trung tâm
    "HaNoi": (50, 3),
    "Hue": (2.85, 0.9),
    "SaiGon": (1.75, -2.4),
    "CanTho": (-1.75, -2.4),
    "HaiPhong": (-2.85, 0.9)
}

# 3. BFS Implementation
def bfs(start, goal, graph):
    frontier = deque([[start]])
    explored = set()
    while frontier:
        path = frontier.popleft()
        node = path[-1]
        if node == goal:
            return path
        if node not in explored:
            for neighbor in graph.neighbors(node):
                new_path = list(path)
                new_path.append(neighbor)
                frontier.append(new_path)
            explored.add(node)
    return None

# 4. DFS Implementation
def dfs(start, goal, graph):
    frontier = [[start]]
    explored = set()
    while frontier:
        path = frontier.pop()
        node = path[-1]
        if node == goal:
            return path
        if node not in explored:
            for neighbor in graph.neighbors(node):
                new_path = list(path)
                new_path.append(neighbor)
                frontier.append(new_path)
            explored.add(node)
    return None

# 5. A* Implementation
def heuristic(node, goal, pos):
    x1, y1 = pos[node]
    x2, y2 = pos[goal]
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def astar(start, goal, graph, pos):
    frontier = [(0, [start])]  # (f, path)
    explored = set()
    g_cost = {start: 0}

    while frontier:
        f, path = heapq.heappop(frontier)
        node = path[-1]

        if node == goal:
            return path

        if node not in explored:
            for neighbor in graph.neighbors(node):
                new_g = g_cost[node] + 1  # mỗi cạnh = 1
                if neighbor not in g_cost or new_g < g_cost[neighbor]:
                    g_cost[neighbor] = new_g
                    h = heuristic(neighbor, goal, pos)
                    f = new_g + h
                    new_path = list(path)
                    new_path.append(neighbor)
                    heapq.heappush(frontier, (f, new_path))
            explored.add(node)
    return None

# 6. Greedy Best-First Search Implementation
def greedy_best_first(start, goal, graph, pos):
    frontier = [(heuristic(start, goal, pos), [start])]  # (h, path)
    explored = set()

    while frontier:
        h, path = heapq.heappop(frontier)  # lấy node có h nhỏ nhất
        node = path[-1]

        if node == goal:
            return path

        if node not in explored:
            for neighbor in graph.neighbors(node):
                if neighbor not in explored:  # tránh trùng lặp
                    new_path = list(path)
                    new_path.append(neighbor)
                    h_neighbor = heuristic(neighbor, goal, pos)
                    heapq.heappush(frontier, (h_neighbor, new_path))
            explored.add(node)
    return None

# 7. Utility functions
def print_path(path):
    if path:
        print(" -> ".join(path))
        print("Path with *:", " * ".join(path))
    else:
        print("No path found")

def draw_path(graph, path, title):
    layout = nx.spring_layout(graph, seed=42)
    nx.draw(graph, layout, with_labels=True, node_size=2000,
            node_color="red", font_color="white")
    if path:
        edge_path = list(zip(path, path[1:]))
        nx.draw_networkx_edges(graph, layout, edgelist=edge_path,
                               edge_color="blue", width=3)
    plt.title(title)
    plt.show()

# 8. Test algorithms
start, goal = "HaNoi", "CanTho"

print("BFS result:")
print_path(bfs(start, goal, G))
draw_path(G, bfs(start, goal, G), "BFS Path")

print("\nDFS result:")
print_path(dfs(start, goal, G))
draw_path(G, dfs(start, goal, G), "DFS Path")

print("\nA* result:")
print_path(astar(start, goal, G, pos))
draw_path(G, astar(start, goal, G, pos), "A* Path")

print("\nGreedy Best-First result:")
print_path(greedy_best_first(start, goal, G, pos))
draw_path(G, greedy_best_first(start, goal, G, pos), "Greedy Best-First Path")