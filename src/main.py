import sys

def dijkstra(graph, start):
  # Initialize distances and visited array
  distances = {node: sys.maxsize for node in graph}
  distances[start] = 0
  visited = set()

  while len(visited) < len(graph):
    # Find the node with the minimum distance
    min_distance = sys.maxsize
    min_node = None
    for node in graph:
      if node not in visited and distances[node] < min_distance:
        min_distance = distances[node]
        min_node = node

    # Mark the node as visited
    visited.add(min_node)

    # Update distances of adjacent nodes
    for neighbor, weight in graph[min_node].items():
      new_distance = distances[min_node] + weight
      if new_distance < distances[neighbor]:
        distances[neighbor] = new_distance

  return distances

# Example usage
graph = {
  'A': {'B': 5, 'C': 2},
  'B': {'A': 5, 'C': 1, 'D': 3},
  'C': {'A': 2, 'B': 1, 'D': 6},
  'D': {'B': 3, 'C': 6}
}

start_node = 'A'
distances = dijkstra(graph, start_node)
print(distances)