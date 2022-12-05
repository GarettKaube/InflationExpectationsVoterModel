import random


class Graph:
    # reference: lecture 16
    def __init__(self, n=0, m=50):
        self.verticies = {}

        # set veticies
        for i in range(1, n + 1):
            self.add_vertex(i)

        # add random edge
        self.n = n

        # Determine whether m is correct
        if m < n - 1 or m > n * (n - 1) / 2:
            raise ValueError('m is not in the reasonable range')
        self.m = m
        self.create_random_edge()
        self.matrix = None

    def create_random_edge(self):
        uniq = set()

        for i in range(self.m):
            while True:
                if i < self.n:
                    from_key = i + 1
                else:
                    from_key = random.randint(1, self.n)
                to_key = from_key
                while to_key == from_key:
                    to_key = random.randint(1, self.n)
                if (to_key, from_key) in uniq or (from_key, to_key) in uniq:
                    continue
                uniq.add((to_key, from_key))
                self.add_edge(from_key, to_key, 1)
                break

    # add a vertex object to the vertices
    def add_vertex(self, vertex):
        self.verticies[vertex] = Vertex(vertex)

    def get_vertex(self, key):
        return self.verticies.get(key)

    def add_edge(self, from_vertex, to_vertex, w):

        # set the neighbor of from_key with weigh
        self.get_vertex(from_vertex).add_neighbor(to_vertex, w)

        # the from_key is also the to_key of to_key with the same weight
        self.get_vertex(to_vertex).add_neighbor(from_vertex, w)

    def print_graph(self):
        for i in range(1, self.n + 1):
            print(self.verticies[i])
        print("______test of random graph end______")

        matrix = [[0 for i in range(self.n)] for i in range(self.n)]
        for i in range(1, self.n+1):
            for j in self.verticies[i].get_neighbors():
                matrix[i-1][j-1] = 1
        print('______print the relation matrix_______')
        for row in matrix:
            print(row)
        return None

    def print_vector(self):
        matrix = self.get_relation_matrix()
        with open('data.txt', 'r') as f:
            vector = [float(i) for i in f.read().split('\n')]
        print('______print the vector_______')
        for i, item in enumerate(vector):
            print(f"value{i+1}:", item)

        res = [0] * self.n
        for i in range(self.n):
            count = 0
            values = 0
            for j in range(self.n):
                if matrix[i][j] != 0:
                    values += vector[j]
                    count += 1
            res[i] = values / count

        print('\n______print the vector_______')
        for i, ele in enumerate(res):
            print(f'value{i+1}:', ele)
    
    def get_relation_matrix(self):
        self.matrix = [[0 for i in range(self.n)] for i in range(self.n)]
        for i in range(1, self.n+1):
            for j in self.verticies[i].get_neighbors():
                self.matrix[i-1][j-1] = 1
        return self.matrix

class Vertex:
    # reference: lecture 16
    def __init__(self, key):
        self.key = key
        self.neighbors = {}

    # set the neighbor of a vertex
    def add_neighbor(self, neighbor, weight):
        self.neighbors[neighbor] = weight

    def get_neighbors(self):
        return list(self.neighbors.keys())

    def __str__(self):
        return f'{self.key} neighbors: {[x for x in self.neighbors.keys()]}'


def main():
    print("\n______test of random graph start______")
    n = 10
    m = 25
    graph = Graph(n, m)
    for i in range(1, n + 1):
        print(graph.verticies[i])
    graph.print_vector()


if __name__ == '__main__':
    main()
