#include <iostream>
#include <vector>
#include <string>
#include <ctime>
#include <cstdlib>
#include <algorithm>

using namespace std;

enum class Direction { UP, RIGHT, DOWN, LEFT };

struct Cell {
    bool wumpus = false;
    bool pit = false;
    bool gold = false;
    bool stench = false;
    bool breeze = false;
    bool glitter = false;
};

struct AgentState {
    int x, y;
    Direction dir;
    bool has_gold = false;
    bool has_arrow = true;
    bool is_alive = true;
    bool escaped = false;
    int score = 0;
};

class Board {
private:
    int size;
    vector<vector<Cell>> grid;
    bool wumpus_alive = true;

    bool is_valid(int x, int y) {
        return x >= 0 && x < size && y >= 0 && y < size;
    }

    void add_percepts() {
        int dx[] = {0, 0, 1, -1};
        int dy[] = {1, -1, 0, 0};

        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                if (grid[i][j].wumpus) {
                    for (int k = 0; k < 4; ++k) {
                        int ni = i + dx[k], nj = j + dy[k];
                        if (is_valid(ni, nj)) grid[ni][nj].stench = true;
                    }
                }
                if (grid[i][j].pit) {
                    for (int k = 0; k < 4; ++k) {
                        int ni = i + dx[k], nj = j + dy[k];
                        if (is_valid(ni, nj)) grid[ni][nj].breeze = true;
                    }
                }
                if (grid[i][j].gold) {
                    grid[i][j].glitter = true;
                }
            }
        }
    }

public:
    Board(int n = 4) : size(n), grid(n, vector<Cell>(n)) {
        srand(time(0));
        // Place Gold
        int gx, gy;
        do { gx = rand() % size; gy = rand() % size; } while (gx == 0 && gy == 0);
        grid[gx][gy].gold = true;

        // Place Wumpus
        int wx, wy;
        do { wx = rand() % size; wy = rand() % size; } while ((wx == 0 && wy == 0) || (wx == gx && wy == gy));
        grid[wx][wy].wumpus = true;

        // Place 3 Pits
        int pits = 0;
        while (pits < 3) {
            int px = rand() % size, py = rand() % size;
            if ((px == 0 && py == 0) || grid[px][py].pit || grid[px][py].wumpus || grid[px][py].gold) continue;
            grid[px][py].pit = true;
            pits++;
        }

        add_percepts();
    }

    Cell get_cell(int x, int y) { return grid[x][y]; }
    int get_size() { return size; }
    bool is_wumpus_alive() { return wumpus_alive; }
    void kill_wumpus() { 
        wumpus_alive = false; 
        for(int i=0; i<size; ++i) 
            for(int j=0; j<size; ++j) 
                grid[i][j].stench = false;
    }
    void remove_gold(int x, int y) { grid[x][y].gold = false; grid[x][y].glitter = false; }
};

class Game {
private:
    Board board;
    AgentState agent;
    vector<vector<bool>> visited;
    vector<vector<string>> agent_map;

    void update_map() {
        visited[agent.x][agent.y] = true;
        Cell current = board.get_cell(agent.x, agent.y);
        string info = "";
        if (current.stench) info += "S";
        if (current.breeze) info += "B";
        if (current.glitter) info += "G";
        if (info == "") info = ".";
        agent_map[agent.x][agent.y] = info;
    }

    void display() {
        cout << "\n--- Agent's Map Knowledge ---\n";
        for (int j = 3; j >= 0; --j) {
            cout << j + 1 << " | ";
            for (int i = 0; i < 4; ++i) {
                if (agent.x == i && agent.y == j) {
                    char arrow;
                    if (agent.dir == Direction::UP) arrow = '^';
                    else if (agent.dir == Direction::RIGHT) arrow = '>';
                    else if (agent.dir == Direction::DOWN) arrow = 'v';
                    else arrow = '<';
                    cout << "[" << arrow << "]\t";
                } else if (visited[i][j]) {
                    cout << " " << agent_map[i][j] << " \t";
                } else {
                    cout << " ? \t";
                }
            }
            cout << endl;
        }
        cout << "  -----------------------------\n";
        cout << "      1 \t2 \t3 \t4\n\n";

        cout << "Simulator: You are at position [" << agent.x + 1 << "," << agent.y + 1 << "] ";
        Cell c = board.get_cell(agent.x, agent.y);
        bool nothing = true;
        if (c.stench || c.breeze || c.glitter) {
            cout << "and you perceive: ";
            if (c.stench) { cout << "a Stench "; nothing = false; }
            if (c.breeze) { cout << (nothing ? "a Breeze " : "and a Breeze "); nothing = false; }
            if (c.glitter) { cout << (nothing ? "a Glitter " : "and a Glitter "); nothing = false; }
        } else {
            cout << "and you perceive nothing.";
        }
        cout << "\nScore: " << agent.score << " | Arrow: " << (agent.has_arrow ? "1" : "0") << " | Gold: " << (agent.has_gold ? "Yes" : "No") << endl;
    }

public:
    Game() : visited(4, vector<bool>(4, false)), agent_map(4, vector<string>(4, "?")) {
        agent.x = 0; agent.y = 0;
        agent.dir = Direction::RIGHT;
        update_map();
    }

    void run() {
        string cmd;
        while (agent.is_alive && !agent.escaped) {
            display();
            cout << "Action (move, left, up, down, right, grab, shoot, climb): ";
            cin >> cmd;
            agent.score--;

            if (cmd == "move") {
                int nx = agent.x, ny = agent.y;
                if (agent.dir == Direction::UP) ny++;
                else if (agent.dir == Direction::RIGHT) nx++;
                else if (agent.dir == Direction::DOWN) ny--;
                else nx--;

                if (nx >= 0 && nx < 4 && ny >= 0 && ny < 4) {
                    agent.x = nx; agent.y = ny;
                    Cell c = board.get_cell(agent.x, agent.y);
                    if (c.pit || (c.wumpus && board.is_wumpus_alive())) {
                        agent.is_alive = false;
                        agent.score -= 100;
                        if (c.pit) cout << "Aaaah! You fell into a pit!\n";
                        else cout << "CHOMP! The Wumpus ate you!\n";
                    }
                    update_map();
                } else {
                    cout << "BUMP! You hit a wall.\n";
                }
            } else if (cmd == "left") {
                agent.dir = Direction::LEFT;
            } else if(cmd == "up") {
                agent.dir = Direction::UP;
            } else if(cmd == "down") {
                agent.dir = Direction::DOWN;
            }else if (cmd == "right") {
                agent.dir = Direction::RIGHT;
            } else if (cmd == "grab") {
                if (board.get_cell(agent.x, agent.y).gold) {
                    agent.has_gold = true;
                    board.remove_gold(agent.x, agent.y);
                    cout << "You picked up the gold!\n";
                } else {
                    cout << "Nothing here to grab.\n";
                }
            } else if (cmd == "shoot") {
                if (agent.has_arrow) {
                    agent.has_arrow = false;
                    cout << "Zing! Arrow shot.\n";
                    int sx = agent.x, sy = agent.y;
                    bool hit = false;
                    while (sx >= 0 && sx < 4 && sy >= 0 && sy < 4) {
                        if (board.get_cell(sx, sy).wumpus && board.is_wumpus_alive()) {
                            hit = true;
                            break;
                        }
                        if (agent.dir == Direction::UP) sy++;
                        else if (agent.dir == Direction::RIGHT) sx++;
                        else if (agent.dir == Direction::DOWN) sy--;
                        else sx--;
                    }
                    if (hit) {
                        cout << "SCREECH!!! You killed the Wumpus!\n";
                        board.kill_wumpus();
                        agent.score += 50;
                    }
                } else {
                    cout << "No arrows left!\n";
                }
            } else if (cmd == "climb") {
                if (agent.x == 0 && agent.y == 0) {
                    agent.escaped = true;
                    if (agent.has_gold) {
                        agent.score += 50;
                        cout << "You escaped with the gold! VICTORY!\n";
                    } else {
                        cout << "You escaped but forgot the gold... MISSION FAILED.\n";
                    }
                } else {
                    cout << "You can only climb out at the start [0,0].\n";
                }
            }
        }
        cout << "Final Score: " << agent.score << endl;
    }
};

int main() {
    Game game;
    game.run();
    return 0;
}
