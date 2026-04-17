import implementation as imp

def custom_draw_tile(graph, id, style):
    r = " . "
    if 'explored' in style and id in style['explored']:
        r = " * "
    if 'path_dict' in style and id in style['path_dict']:
        next_node = style['path_dict'][id]
        if next_node:
            (x1, y1) = id
            (x2, y2) = next_node
            if x2 == x1 + 1: r = " > "
            if x2 == x1 - 1: r = " < "
            if y2 == y1 + 1: r = " v "
            if y2 == y1 - 1: r = " ^ "
    if 'start' in style and id == style['start']: r = " A "
    if 'goal' in style and id == style['goal']:   r = " Z "
    if id in graph.walls: r = "###"
    return r

imp.draw_tile = custom_draw_tile

def main():
    # Aumentando o grid para 25x10
    g = imp.GridWithWeights(25, 10)
    
    walls = []
    # Parede vertical (fundo da armadilha)
    for y in range(2, 8): walls.append((12, y))
    # Paredes horizontais (teto e chão da armadilha)
    for x in range(7, 13): walls.append((x, 2))
    for x in range(7, 13): walls.append((x, 7))
    g.walls = walls

    g.weights = {}

    start = (2, 4)  # Meio-esquerda
    goal = (22, 4)  # Meio-direita

    came_from, cost_so_far = imp.a_star_search(g, start, goal)
    path = imp.reconstruct_path(came_from, start, goal)

    path_dict = {}
    if path:
        for i in range(len(path) - 1):
            path_dict[path[i]] = path[i+1]
        path_dict[path[-1]] = None

    print("\n=== Resultado da Busca A* (Cenário Armadilha) ===")
    imp.draw_grid(g, explored=cost_so_far, path_dict=path_dict, start=start, goal=goal)
    print(f"\nCusto total do caminho: {cost_so_far.get(goal, 'Caminho não encontrado')}")
    print(f"Total de nós explorados: {len(cost_so_far)}\n")

if __name__ == '__main__':
    main()
