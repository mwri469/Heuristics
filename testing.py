import numpy as np
import timeit

def main():
    n, F, D = create_small_instance()

    print(compute_cost_change([0,1,2,3,4,5], F, D, 1, 4)[0])
    print(compute_cost([0,4,2,3,1,5], F, D) - compute_cost([0,1,2,3,4,5], F, D))

def steepestdescent_to_local_minimum(x, F, D):
    """
    Repeatedly apply our neighbourhood steepest descent until we stop at a local minimum.
    Args:
        x             : Current solution (x[i] is the location index for facility i)
        F (np.ndarray): Flow matrix.
        D (np.ndarray): Distance matrix.
        plot_manager  : An optional plot manager to collect data for plotting
    Returns:
        x (list): Best solution found.
        cost (float): Objective function value of solution x
    """
    cost = compute_cost(x, F, D)
    # TODO Fix this code
    n = len(x)
    while True:
        # Get all neighbours by finding possible swaps
        neighbours = [(compute_cost_change(x, F, D, i, j)) 
                       for j in range(1,n)
                       for i in range (0,n-1)]

        # Get the minimum of the neighbours, test for local minimum
        best_neighbour = (np.inf, [])

        for neighbour in neighbours:
            if neighbour[0] < best_neighbour[0]:
                best_neighbour = neighbour

        if best_neighbour[0] >= 0:
            # Local minimum has been reached
            break
        else:
            # Add to new cost
            x = best_neighbour[1].copy()
            cost += best_neighbour[0]
            
        
    return x, cost

def timed(F, D, x):
    randidx = np.random.randint(0, 99, dtype=int)
    x_new = swap(x.copy(), 99-randidx, randidx)

    start = timeit.timeit()
    cost = compute_cost(x_new, F, D) - compute_cost(x, F, D)
    end = timeit.timeit()

    print(f'Cost is: {cost}, took: {end - start}')

    start = timeit.timeit()
    cost = compute_cost_change(x, F, D, 2, 7)
    end = timeit.timeit()

    print(f'Cost is: {cost[0]}, took: {end - start}')  

def swap(x, i, j):
    """
    Swap the locations of facilities i and j in the solution x.
    Args:
        x (list): Permutation of facilities.
        i (int): Index of facility i.
        j (int): Index of facility j.
    Returns:
        list: Updated solution/permutation after swapping.
    """
    x[i], x[j] = x[j], x[i]
    return x

def compute_cost(x, F, D):
    """
    Compute the cost of a solution (permutation) x in which facility i is in location x[i]
    Args:
        x (list): Permutation of facilities, i.e. x[i] gives the index of the location of facility i.
        F (np.ndarray): Flow matrix: F[i][j] is the flow from facility i to facility j.
        D (np.ndarray): Distance matrix: D[p,q] is the distance between locations p and q.
    Returns:
        float: Total cost of the permutation.
    """
    n = len(x)
    cost = 0.0
    for i in range(n):
        for j in range(n):
            cost += F[i, j] * D[x[i], x[j]]
    return cost

def new_compute_cost(x, F, D, p, q):
    """
    Compute the cost of a solution (permutation) x in which facility i is in location x[i]
    Args:
        x (list): Permutation of facilities, i.e. x[i] gives the index of the location of facility i.
        F (np.ndarray): Flow matrix: F[i][j] is the flow from facility i to facility j.
        D (np.ndarray): Distance matrix: D[p,q] is the distance between locations p and q.
    Returns:
        float: Total cost of the permutation.
    """
    n = len(x)
    cost = 0.0
    for i in range(n):
        for j in range(n):
            if i==j:
                continue
            if j==p or j==q or i==p or i==q:
                cost += F[i, j] * D[x[i], x[j]]
            else:
                continue
                
    return cost

def nextdescent_to_local_minimum(x, F, D):
    """
    Repeatedly sweep our neighbourhood, keeping improvements as we find them, until we are at a local minimum
    Args:
        x             : Current solution (x[i] is the index of the location for facility i)
        F (np.ndarray): Flow matrix.
        D (np.ndarray): Distance matrix.
        plot_manager  : An optional plot manager to collect data for plotting
    Returns:
        x (list): Best solution found.
        cost (float): Objective function value of solution x
    """
    cost = compute_cost(x, F, D)
    # if plot_manager: plot_manager.initialise_new_run("Next Descent")
    n = len(x)

    # TODO FIX THE FOLLOWING CODE
    for p in range(n-1):
        for q in range(p+1, n):
            change_in_cost, x_test = compute_cost_change(x,F,D,p,q)
            print(f'Iteration: {(p*n) + q + 1}, x = {x}, x* = {x_test}')
            if change_in_cost < 0:
                # If we get an improvement in cost, keep the solution
                x = x_test.copy()

    return x, cost

def compute_cost_change(x, F, D, q, r):
    """
    Compute the change in cost if we change a solution x by swapping the q'th and r'th entries.
    Args:
        x (list): solution, i.e. the location assigned to each facility
        F (np.ndarray): Flow matrix.
        D (np.ndarray): Distance matrix.
        q,r: The indices of the entries (i.e. facilities) in x that are swapped (i.e swap locations)
    Returns:
        cost_change: The change in the solution objective function value when going from x to x_test
        x_test (list): The new solution (with the q'th and r'th entries swapped)
    """
    n = len(x)
    x_test = swap(x.copy(), q, r) # This is the new solution we wish to quickly evaluate the objective change for
    #TODO: Make this efficient
    cost1 = 0.0
    cost2 = 0.0
    for i in range(n):
        for j in range(n):
            if i==j:
                continue
            if j==r or j==q or i==r or i==q:
                cost1 += F[i, j] * D[x[i], x[j]]
                cost2 += F[i,j] * D[x_test[i], x_test[j]]
            else:
                continue

    cost_change = cost2 - cost1

    return cost_change, x_test

def create_small_instance():
    """
    Create a small problem instance.
    Returns:
        n = number of facilities
        F = flow between each facility
        D = distance between each location
    """
    n = 6  # Number of facilities
    # Distance between each pair of locations
    D=[[0,2,61,50,3,8,],
    [45,0,79,88,68,75,],
    [93,62,0,67,57,23,],
    [97,25,23,0,100,72,],
    [72,39,80,67,0,38,],
    [19,70,83,7,66,0,],]
    D=np.array(D)
    # Flow between each facility
    F=[[0,81,83,2,15,34,],
    [65,0,43,53,14,8,],
    [87,22,0,72,26,20,],
    [34,1,74,0,81,53,],
    [73,53,42,78,0,93,],
    [33,68,89,94,41,0,],]
    F=np.array(F)

    # Updated 14/3/2024 to return "n,F,D", not "n,D,F".
    # F and D happen to be interchangeable for this problem, but this code is easier to 
    # match with the spreadsheet given on Canvas with this correctio n.
    # This change does not make any difference to the way your code runs (nor the objective functin values calculated)
    # return n,D,F
    return n, F, D

if __name__ == '__main__':
    main()