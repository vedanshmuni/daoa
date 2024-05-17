# daoa
PRIMS ALGO:

#include <stdio.h>
#include <limits.h>

#define MAX_VERTICES 100

int minKey(int key[], int mstSet[], int V) {
    int min = INT_MAX, min_index;
    for (int v = 0; v < V; v++)
        if (mstSet[v] == 0 && key[v] < min)
            min = key[v], min_index = v;
    return min_index;
}

void primMST(int graph[MAX_VERTICES][MAX_VERTICES], int V) {
    int parent[V], key[V], mstSet[V];
    for (int i = 0; i < V; i++)
        key[i] = INT_MAX, mstSet[i] = 0;
    key[0] = 0, parent[0] = -1;

    for (int count = 0; count < V - 1; count++) {
        int u = minKey(key, mstSet, V);
        mstSet[u] = 1;
        for (int v = 0; v < V; v++)
            if (graph[u][v] && mstSet[v] == 0 && graph[u][v] < key[v])
                parent[v] = u, key[v] = graph[u][v];
    }

    printf("Minimum Spanning Tree (MST) for the given graph:\n");
    int minCost = 0;
    for (int i = 1; i < V; i++) {
        printf("%d - %d \t%d\n", parent[i], i, graph[i][parent[i]]);
        minCost += graph[i][parent[i]];
    }
    printf("Total cost of MST: %d\n", minCost);
}

int main() {
    int V;
    printf("Enter the number of vertices in the graph: ");
    scanf("%d", &V);

    int graph[MAX_VERTICES][MAX_VERTICES];
    printf("Enter the adjacency matrix of the graph:\n");
    for (int i = 0; i < V; i++)
        for (int j = 0; j < V; j++)
            scanf("%d", &graph[i][j]);

    primMST(graph, V);
    return 0;
}


KRUSKALS:

#include <stdio.h>
#include <stdlib.h>

#define MAX_VERTICES 100
#define MAX_EDGES 1000

int edges[MAX_EDGES][3];
int numEdges = 0;

int parent[MAX_VERTICES];

void makeSet(int n)
{
    for (int i = 0; i < n; i++)
        parent[i] = -1;
}

int findSet(int x)
{
    if (parent[x] == -1)
        return x;
    return findSet(parent[x]);
}

void unionSet(int x, int y)
{
    int xRoot = findSet(x);
    int yRoot = findSet(y);
    parent[xRoot] = yRoot;
}

int cmpEdge(const void *a, const void *b)
{
    int *edgeA = (int *)a;
    int *edgeB = (int *)b;
    return edgeA[2] - edgeB[2];
}

void kruskalMST(int numVertices)
{
    makeSet(numVertices);

    qsort(edges, numEdges, sizeof(edges[0]), cmpEdge);

    int count = 0, totalWeight = 0;
    printf("Edges in the MST:\n");

    for (int i = 0; i < numEdges; i++)
    {
        int src = findSet(edges[i][0]);
        int dest = findSet(edges[i][1]);

        if (src != dest)
        {
            unionSet(src, dest);
            printf("%d -- %d\n", edges[i][0], edges[i][1]);
            totalWeight += edges[i][2];
            count++;
        }

        if (count == numVertices - 1)
            break;
    }

    printf("Total weight of MST: %d\n", totalWeight);
}

int main()
{
    int numVertices, numEdges, i;

    printf("Enter the number of vertices: ");
    scanf("%d", &numVertices);

    printf("Enter the number of edges: ");
    scanf("%d", &numEdges);

    printf("Enter the edges (source, destination, weight):\n");
    for (i = 0; i < numEdges; i++)
    {
        int src, dest, weight;
        scanf("%d %d %d", &src, &dest, &weight);
        edges[i][0] = src;
        edges[i][1] = dest;
        edges[i][2] = weight;
    }

    kruskalMST(numVertices);

    return 0;
}


KNAPSACK(GREEDY):

#include <stdio.h>

void swap(int *a, int *b)
{
	int temp = *a;
	*a = *b;
	*b = temp;
}

void bubbleSort(int weights[], int values[], double ratios[], int n)
{
	int i, j;
	for (i = 0; i < n - 1; i++)
	{
		for (j = 0; j < n - i - 1; j++)
		{
			if (ratios[j] < ratios[j + 1])
			{
				swap(&weights[j], &weights[j + 1]);
				swap(&values[j], &values[j + 1]);
				swap(&ratios[j], &ratios[j + 1]);
			}
		}
	}
}

float fractionalKnapsack(int capacity, int weights[], int values[], int n)
{
	float totalValue = 0.0;
	float ratio[n];

	for (int i = 0; i < n; i++)
	{
		ratio[i] = (float)values[i] / weights[i];
	}

	bubbleSort(weights, values, ratio, n);

	for (int i = 0; i < n; i++)
	{
		if (weights[i] <= capacity)
		{
			totalValue += values[i];
			capacity -= weights[i];
		}
		else
		{
			totalValue += ratio[i] * capacity;
			break;
		}
	}
} 

COIN CHANGE GREEDY:

#include <stdio.h>
#define COINS 8
#define MAX 20

// All denominations of Indian Currency
int coins[COINS] = {1, 2, 5, 10, 20,
                    50, 100, 500};

void coinChange(int cost)
{
    int coinList[MAX] = {0};
    int i, k = 0;

    for (i = COINS - 1; i >= 0; i--)
    {
        while (cost >= coins[i])
        {
            cost -= coins[i];
            // Add coin in the list
            coinList[k++] = coins[i];
        }
    }

    for (i = 0; i < k; i++)
    {
        // Print
        printf("%d ", coinList[i]);
    }
    return;
}

int main(void)
{
    // input value
    int n = 93;

    printf("Following is minimal number"
           "of change for %d: ",
           n);
    coinChange(n);
    return 0;
}


DJISKTRA GREEDY

#include <stdio.h>
#define INFINITY 9999
#define MAX 10

void Dijkstra(int graph[MAX][MAX], int numVertices, int start);

void Dijkstra(int graph[MAX][MAX], int numVertices, int start)
{
    int cost[MAX][MAX], distance[MAX], predecessor[MAX];
    int visited[MAX], count, minDistance, nextNode, i, j;

    // Creating cost matrix
    for (i = 0; i < numVertices; i++)
        for (j = 0; j < numVertices; j++)
            if (graph[i][j] == 0)
                cost[i][j] = INFINITY;
            else
                cost[i][j] = graph[i][j];

    for (i = 0; i < numVertices; i++)
    {
        distance[i] = cost[start][i];
        predecessor[i] = start;
        visited[i] = 0;
    }

    distance[start] = 0;
    visited[start] = 1;
    count = 1;

    while (count < numVertices - 1)
    {
        minDistance = INFINITY;

        for (i = 0; i < numVertices; i++)
            if (distance[i] < minDistance && !visited[i])
            {
                minDistance = distance[i];
                nextNode = i;
            }

        visited[nextNode] = 1;
        for (i = 0; i < numVertices; i++)
            if (!visited[i])
                if (minDistance + cost[nextNode][i] < distance[i])
                {
                    distance[i] = minDistance + cost[nextNode][i];
                    predecessor[i] = nextNode;
                }
        count++;
    }

    // Printing the distance
    for (i = 0; i < numVertices; i++)
        if (i != start)
        {
            printf("\nDistance from source to %d: %d", i, distance[i]);
        }
}

int main()
{
    int graph[MAX][MAX], i, j, numVertices, startVertex;
    numVertices = 4;

    graph[0][0] = 0;
    graph[0][1] = 1;
    graph[0][2] = 3;
    graph[0][3] = 0;

    graph[1][0] = 0;
    graph[1][1] = 0;
    graph[1][2] = 1;
    graph[1][3] = 10;

    graph[2][0] = 1;
    graph[2][1] = 0;
    graph[2][2] = 0;
    graph[2][3] = 4;

    graph[3][0] = 0;
    graph[3][1] = 0;
    graph[3][2] = 0;
    graph[3][3] = 0;

    startVertex = 0;
    Dijkstra(graph, numVertices, startVertex);

    return 0;
}

MATRIX CHAIN DP:

#include <stdio.h>
#include <limits.h>

int main()
{
    int num;
    printf("Suppose you have 3 Matrices of size 2x5, 5x7, 7x9\n");
    printf("Enter the input as 3 5 7 9\n");
    printf("Enter the total number of matrices:\n");
    scanf("%d", &num);
    printf("Enter the matrix array:\n");
    int arr[num + 1];
    for (int i = 0; i < num + 1; i++)
    {
        scanf("%d", &arr[i]);
    }
    int n = num + 1;
    int m[n][n];
    int i, j, k, L, q;
    for (i = 1; i < n; i++)
        m[i][i] = 0;
    for (L = 2; L < n; L++)
    {
        for (i = 1; i < n - L + 1; i++)
        {
            j = i + L - 1;
            if (j == n)
                continue;
            m[i][j] = INT_MAX;
            for (k = i; k <= j - 1; k++)
            {
                q = m[i][k] + m[k + 1][j] + arr[i - 1] * arr[k] * arr[j];
                if (q < m[i][j])
                {
                    m[i][j] = q;
                }
            }
        }
    }
    int min = m[1][n - 1];
    printf("Minimum number of multiplications is: %d\n", min);
    return 0;
} 


COIN CHANGE DP:

#include <stdio.h>
#include <math.h>
#include <conio.h>

int coinChange(int coins[], int n, int amount)
{
    int minCoins[amount + 1];

    for (int i = 0; i < n; i++)
    {
        minCoins[i] = amount + 1;
    }
    minCoins[0] = 0;

    for (int i = 1; i <= amount; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (coins[j] <= i)
            {
                int subres = minCoins[i - coins[j]];
                if (subres != amount + 1)
                {
                    minCoins[i] = fmin(minCoins[i], subres + 1);
                }
            }
        }
    }

    if (minCoins[amount] == amount + 1)
    {
        return -1;
    }
    else
    {
        return minCoins[amount];
    }
};

int main()
{
    int n;
    printf("Enter the number of denomination: ");
    scanf("%d", &n);

    int denomination[n];
    printf("Enter the denomination: \n");
    for (int i = 0; i < n; i++)
    {
        scanf("%d", &denomination[i]);
    }

    int targetAmount;
    printf("Enter the amount: ");
    scanf("%d", &targetAmount);

    int minCoins = coinChange(denomination, n, targetAmount);

    if (minCoins == -1)
    {
        printf("not possible");
    }
    else
    {
        printf("required coins: %d", minCoins);
    }
    return 0;
} 

TSP DP:

#include <stdio.h>
#include <limits.h>

#define MAX_CITIES 10

int tsp(int cities, int graph[MAX_CITIES][MAX_CITIES], int start, int mask)
{
    int min_cost = INT_MAX;
    int visited_all = (1 << cities) - 1;

    if (mask == visited_all)
    {
        return graph[start][0];
    }

    for (int city = 0; city < cities; city++)
    {
        if (!(mask & (1 << city)))
        {
            int new_mask = mask | (1 << city);
            int cost = graph[start][city] + tsp(cities, graph, city, new_mask);
            min_cost = (cost < min_cost) ? cost : min_cost;
        }
    }

    return min_cost;
}

int main()
{
    int graph[MAX_CITIES][MAX_CITIES] = {
        {0, 22, 26, 30},
        {30, 0, 45, 35},
        {25, 45, 0, 60},
        {30, 35, 40, 0}};
    int cities = 4;
    int start = 0;

    int min_cost = tsp(cities, graph, start, 1);
    printf("Minimum cost: %d\n", min_cost);

    return 0;
} 



LCS DP:


#include <stdio.h>
#include <string.h>

int lcs(char *X, char *Y, int m, int n)
{
    int dp[m + 1][n + 1];

    // Initialize the dp array
    for (int i = 0; i <= m; i++)
    {
        for (int j = 0; j <= n; j++)
        {
            if (i == 0 || j == 0)
                dp[i][j] = 0;
            else if (X[i - 1] == Y[j - 1])
                dp[i][j] = dp[i - 1][j - 1] + 1;
            else
                dp[i][j] = (dp[i - 1][j] > dp[i][j - 1]) ? dp[i - 1][j] : dp[i][j - 1];
        }
    }

    return dp[m][n];
}

int main()
{
    char X[] = "STONE";
    char Y[] = "LONGEST";

    int m = strlen(X);
    int n = strlen(Y);

    printf("Length of LCS: %d\n", lcs(X, Y, m, n));

    return 0;
} 


N QUEEN BACKTRACKING:

#include <stdio.h>
#include <stdbool.h>

#define N 4

bool isSafe(int board[N][N], int row, int col)
{
    for (int i = 0; i < col; i++)
        if (board[row][i])
            return false;

    for (int i = row, j = col; i >= 0 && j >= 0; i--, j--)
        if (board[i][j])
            return false;

    for (int i = row, j = col; i < N && j >= 0; i++, j--)
        if (board[i][j])
            return false;

    return true;
}

bool solveNQueens(int board[N][N], int col)
{
    if (col == N)
        return true;

    for (int i = 0; i < N; i++)
    {
        if (isSafe(board, i, col))
        {
            board[i][col] = 1;

            if (solveNQueens(board, col + 1))
                return true;

            board[i][col] = 0;
        }
    }

    return false;
}

int main()
{
    int board[N][N] = {0};

    if (solveNQueens(board, 0))
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
                printf("%d ", board[i][j]);
            printf("\n");
        }
    else
        printf("Solution does not exist\n");

    return 0;
}

SUM OF SUBSETS BACKTRACKING:

#include <stdio.h>

#include <stdio.h>

void printSubset(int arr[], int n, int sum, int subset[], int subsetSize, int currentIndex)
{
    if (sum == 0)
    {
        for (int i = 0; i < n; i++)
        {
            if (subset[i] == 1)
            {
                printf("1 ");
            }
            else
            {
                printf("0 ");
            }
        }
        printf("\n");
        return;
    }

    if (currentIndex == n)
        return;

    subset[subsetSize] = 1;
    printSubset(arr, n, sum - arr[currentIndex], subset, subsetSize + 1, currentIndex + 1);

    subset[subsetSize] = 0;
    printSubset(arr, n, sum, subset, subsetSize + 1, currentIndex + 1);
}

int main()
{
    int arr[] = {3, 4, 5, 6};
    int n = sizeof(arr) / sizeof(arr[0]);
    int sum = 9;
    int subset[n];

    printf("Subsets with sum %d:\n", sum);
    printSubset(arr, n, sum, subset, 0, 0);

    return 0;
}

JOB SCHEDULING GREEDY:

#include <stdio.h>

#define MAX_JOBS 100

// Function to sort jobs based on profit
void sortJobs(int profit[], int deadline[], int n)
{
    int i, j, temp;
    for (i = 0; i < n - 1; i++)
    {
        for (j = 0; j < n - i - 1; j++)
        {
            if (profit[j] < profit[j + 1])
            {
                // Swap profit values
                temp = profit[j];
                profit[j] = profit[j + 1];
                profit[j + 1] = temp;

                // Swap deadline values
                temp = deadline[j];
                deadline[j] = deadline[j + 1];
                deadline[j + 1] = temp;
            }
        }
    }
}

// Function to find the maximum profit from the given jobs
int maxProfit(int profit[], int deadline[], int n, int job_sequence[])
{
    int slots[MAX_JOBS] = {0}; // Array to store job slots
    int max_profit = 0;
    int job_count = 0;

    // Sort jobs based on profit in descending order
    sortJobs(profit, deadline, n);

    // Iterate through jobs and assign them to available slots
    for (int i = 0; i < n; i++)
    {
        for (int j = deadline[i] - 1; j >= 0; j--)
        {
            if (slots[j] == 0)
            {
                slots[j] = profit[i];
                job_sequence[job_count++] = i + 1; // Store job sequence
                max_profit += profit[i];
                break;
            }
        }
    }

    return max_profit;
}

int main()
{
    int n, i;
    int profit[MAX_JOBS], deadline[MAX_JOBS];
    int job_sequence[MAX_JOBS];

    printf("Enter the number of jobs: ");
    scanf("%d", &n);

    printf("Enter the profit and deadline for each job:\n");
    for (i = 0; i < n; i++)
    {
        printf("Job %d: ", i + 1);
        scanf("%d %d", &profit[i], &deadline[i]);
    }

    int max_prof = maxProfit(profit, deadline, n, job_sequence);
    printf("The maximum profit that can be obtained is: %d\n", max_prof);

    return 0;
}


KMP STRING MATCHING:

#include <stdio.h>
#include <string.h>

void computeLPSArray(char *pat, int M, int *lps)
{
    int len = 0;
    lps[0] = 0; // lps[0] is always 0

    // Calculate lps[] for other positions
    int i = 1;
    while (i < M)
    {
        if (pat[i] == pat[len])
        {
            len++;
            lps[i] = len;
            i++;
        }
        else
        {
            if (len != 0)
            {
                len = lps[len - 1];
            }
            else
            {
                lps[i] = 0;
                i++;
            }
        }
    }
}

int KMPSearch(char *pat, char *txt)
{
    int M = strlen(pat);
    int N = strlen(txt);
    int lps[M];

    // Preprocess the pattern to fill lps[] array
    computeLPSArray(pat, M, lps);

    int i = 0; // Index for txt[]
    int j = 0; // Index for pat[]
    while (i < N)
    {
        if (pat[j] == txt[i])
        {
            j++;
            i++;
        }

        if (j == M)
        {
            // Pattern found at index i - j
            return i - j;
        }
        else if (i < N && pat[j] != txt[i])
        {
            if (j != 0)
            {
                j = lps[j - 1];
            }
            else
            {
                i = i + 1;
            }
        }
    }

    // Pattern not found
    return -1;
}

int main()
{
    char txt[] = "ABABCABCABABABD";
    char pat[] = "ABABD";
    int result = KMPSearch(pat, txt);
    if (result == -1)
    {
        printf("Pattern not found in the text.\n");
    }
    else
    {
        printf("Pattern found at index %d\n", result);
    }
    return 0;
}


15 PUZZLE PROBLEM:

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>

#define N 4

// Helper function to calculate Manhattan distance heuristic
int calculateManhattanDistance(int board[N][N])
{
    int distance = 0;

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int value = board[i][j];
            if (value != 0)
            {
                int targetRow = (value - 1) / N;
                int targetCol = (value - 1) % N;
                distance += abs(i - targetRow) + abs(j - targetCol);
            }
        }
    }

    return distance;
}

// Helper function to print the board state
void printBoard(int board[N][N])
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%d ", board[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Helper function to perform a move on the board
void performMove(int board[N][N], int emptyRow, int emptyCol, int newRow, int newCol, int newBoard[N][N])
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            newBoard[i][j] = board[i][j];
        }
    }
    newBoard[emptyRow][emptyCol] = newBoard[newRow][newCol];
    newBoard[newRow][newCol] = 0;
}

// Solve the 15 puzzle using iterative deepening search
void solve15Puzzle(int initialBoard[N][N])
{
    int board[N][N];
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            board[i][j] = initialBoard[i][j];
        }
    }
    int moves = 0;

    while (calculateManhattanDistance(board) > 0)
    {
        printf("After %d moves:\n", moves);
        printBoard(board);

        int emptyRow = -1;
        int emptyCol = -1;

        // Find the position of the empty space (zero) in the puzzle
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                if (board[i][j] == 0)
                {
                    emptyRow = i;
                    emptyCol = j;
                    break;
                }
            }
        }

        int nextBoard[N][N];
        int minDistance = INT_MAX;

        // Try moving the empty space in all four directions
        int directions[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        for (int k = 0; k < 4; k++)
        {
            int newRow = emptyRow + directions[k][0];
            int newCol = emptyCol + directions[k][1];

            if (newRow >= 0 && newRow < N && newCol >= 0 && newCol < N)
            {
                int newBoard[N][N];
                performMove(board, emptyRow, emptyCol, newRow, newCol, newBoard);
                int distance = calculateManhattanDistance(newBoard);
                if (distance < minDistance)
                {
                    minDistance = distance;
                    for (int i = 0; i < N; i++)
                    {
                        for (int j = 0; j < N; j++)
                        {
                            nextBoard[i][j] = newBoard[i][j];
                        }
                    }
                }
            }
        }

        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                board[i][j] = nextBoard[i][j];
            }
        }
        moves++;
    }

    printf("Solution found in %d moves:\n", moves);
    printBoard(board);
}

int main()
{
    // Example initial puzzle state (1 represents empty space)
    int initialBoard[N][N] = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 0, 11},
        {13, 14, 15, 12}};

    solve15Puzzle(initialBoard);

    return 0;
}

STRASSENS MATRIX MULTIPLICATION:
#include <stdio.h>
#include <stdlib.h>

void add(int n, int matA[n][n], int matB[n][n], int matC[n][n]) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matC[i][j] = matA[i][j] + matB[i][j];
        }
    }
}

void sub(int n, int matA[n][n], int matB[n][n], int matC[n][n]) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matC[i][j] = matA[i][j] - matB[i][j];
        }
    }
}

void multiply(int n, int matA[n][n], int matB[n][n], int matC[n][n]) {
    if (n == 1) {
        matC[0][0] = matA[0][0] * matB[0][0];
    } else {
        int A11[n/2][n/2], A12[n/2][n/2], A21[n/2][n/2], A22[n/2][n/2];
        int B11[n/2][n/2], B12[n/2][n/2], B21[n/2][n/2], B22[n/2][n/2];
        int P1[n/2][n/2], P2[n/2][n/2], P3[n/2][n/2], P4[n/2][n/2], P5[n/2][n/2], P6[n/2][n/2], P7[n/2][n/2];
        int C11[n/2][n/2], C12[n/2][n/2], C21[n/2][n/2], C22[n/2][n/2];

        // Splitting the matrices into 4 sub-matrices
        for (int i = 0; i < n / 2; i++) {
            for (int j = 0; j < n / 2; j++) {
                A11[i][j] = matA[i][j];
                A12[i][j] = matA[i][j + n / 2];
                A21[i][j] = matA[i + n / 2][j];
                A22[i][j] = matA[i + n / 2][j + n / 2];

                B11[i][j] = matB[i][j];
                B12[i][j] = matB[i][j + n / 2];
                B21[i][j] = matB[i + n / 2][j];
                B22[i][j] = matB[i + n / 2][j + n / 2];
            }
        }

        // Calculating P1 to P7:
        add(n/2, A11, A22, P1);
        add(n/2, B11, B22, P2);
        multiply(n/2, P1, P2, P1);

        add(n/2, A21, A22, P2);
        multiply(n/2, P2, B11, P2);

        sub(n/2, B12, B22, P3);
        multiply(n/2, A11, P3, P3);

        sub(n/2, B21, B11, P4);
        multiply(n/2, A22, P4, P4);

        add(n/2, A11, A12, P5);
        multiply(n/2, P5, B22, P5);

        sub(n/2, A21, A11, P6);
        add(n/2, B11, B12, P7);
        multiply(n/2, P6, P7, P6);

        sub(n/2, A12, A22, P7);
        add(n/2, B21, B22, C11);
        multiply(n/2, P7, C11, P7);

        // Calculating C11, C12, C21, C22:
        add(n/2, P1, P4, C11);
        sub(n/2, C11, P5, C11);
        add(n/2, C11, P7, C11);

        add(n/2, P3, P5, C12);

        add(n/2, P2, P4, C21);

        add(n/2, P1, P3, C22);
        sub(n/2, C22, P2, C22);
        add(n/2, C22, P6, C22);

        // Grouping the results obtained in a single matrix:
        for (int i = 0; i < n / 2; i++) {
            for (int j = 0; j < n / 2; j++) {
                matC[i][j] = C11[i][j];
                matC[i][j + n / 2] = C12[i][j];
                matC[i + n / 2][j] = C21[i][j];
                matC[i + n / 2][j + n / 2] = C22[i][j];
            }
        }
    }
}

int main() {
    int n;
    printf("Enter the size of matrices (n x n): ");
    scanf("%d", &n);

    int A[n][n], B[n][n], result[n][n];

    printf("Enter the elements of matrix A:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            scanf("%d", &A[i][j]);
        }
    }

    printf("Enter the elements of matrix B:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            scanf("%d", &B[i][j]);
        }
    }

    multiply(n, A, B, result);

    printf("Resultant Matrix:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", result[i][j]);
        }
        printf("\n");
    }

    return 0;
}

