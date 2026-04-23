from collections import deque
import heapq

#Guilherme Bento Ramos 185226
#Missionarios e Canibais DFS e A*
#Feito com auxilio do Gemini para debug

class State:
    def __init__(self, m_left, c_left, boat, m_right, c_right, parent=None, action=None, cost=0, heuristic=0):
        self.m_left = m_left
        self.c_left = c_left
        self.boat = boat  # 0 for left bank, 1 for right bank
        self.m_right = m_right
        self.c_right = c_right
        self.parent = parent
        self.action = action
        self.cost = cost
        self.heuristic = heuristic

    def is_goal(self):
        return self.m_left == 0 and self.c_left == 0

    def is_valid(self):
        if self.m_left < 0 or self.c_left < 0 or self.m_right < 0 or self.c_right < 0:
            return False
        if (self.m_left > 0 and self.m_left < self.c_left) or (self.m_right > 0 and self.m_right < self.c_right):
            return False
        return True

    def __eq__(self, other):
        return (self.m_left == other.m_left and self.c_left == other.c_left and 
                self.boat == other.boat and self.m_right == other.m_right and 
                self.c_right == other.c_right)

    def __hash__(self):
        return hash((self.m_left, self.c_left, self.boat, self.m_right, self.c_right))

    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)

def get_successors(state):
    successors = []
    moves = [(1, 0), (2, 0), (0, 1), (0, 2), (1, 1)]
    
    if state.boat == 0:  # Boat is on the left
        for m, c in moves:
            new_state = State(state.m_left - m, state.c_left - c, 1,
                              state.m_right + m, state.c_right + c, 
                              parent=state, action=f"Move {m}M and {c}C to right",
                              cost=state.cost + 1)
            if new_state.is_valid():
                successors.append(new_state)
    else:  # Boat is on the right
        for m, c in moves:
            new_state = State(state.m_left + m, state.c_left + c, 0,
                              state.m_right - m, state.c_right - c, 
                              parent=state, action=f"Move {m}M and {c}C to left",
                              cost=state.cost + 1)
            if new_state.is_valid():
                successors.append(new_state)
    return successors

def dfs(initial_state):
    stack = [initial_state]
    visited = set()
    
    while stack:
        state = stack.pop()
        if state.is_goal():
            return state
        
        if state not in visited:
            visited.add(state)
            for successor in get_successors(state):
                stack.append(successor)
    return None

def heuristic(state):
    # Simple heuristic: number of people on the left bank
    return state.m_left + state.c_left

def a_star(initial_state):
    initial_state.heuristic = heuristic(initial_state)
    priority_queue = [initial_state]
    visited = {}
    
    while priority_queue:
        state = heapq.heappop(priority_queue)
        
        if state.is_goal():
            return state
            
        state_key = (state.m_left, state.c_left, state.boat)
        if state_key in visited and visited[state_key] <= state.cost:
            continue
        visited[state_key] = state.cost
        
        for successor in get_successors(successor := state): # dummy for clarity in iteration
            pass # actual iteration below
        
        for successor in get_successors(state):
            successor.heuristic = heuristic(successor)
            heapq.heappush(priority_queue, successor)
    return None

def print_solution(solution):
    if solution is None:
        print("No solution found.")
        return
    
    path = []
    curr = solution
    while curr:
        path.append(curr)
        curr = curr.parent
    
    for i, state in enumerate(reversed(path)):
        if state.action:
            print(f"Step {i}: {state.action}")
        print(f"   Left: {state.m_left}M, {state.c_left}C | Boat: {'Left' if state.boat == 0 else 'Right'} | Right: {state.m_right}M, {state.c_right}C")

if __name__ == "__main__":
    initial_state = State(3, 3, 0, 0, 0)
    
    print("--- Solving with DFS ---")
    dfs_solution = dfs(initial_state)
    print_solution(dfs_solution)
    
    print("\n--- Solving with A* ---")
    astar_solution = a_star(initial_state)
    print_solution(astar_solution)
