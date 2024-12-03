# ENGSCI 760 2024
# Campus (facility) location
# (c) Andrew Mason, University of Auckland
# 4/3/2024 v5b: Changed so "create_small_instance" now returns "n, F, D", not "n, D, F" (which still works and gives the same objective fn value, but is confusing.)

import random
import numpy as np
import plotly.express
import timeit
from challenge_instances import *

class PlotManager:
    """ PlotManager collects and plots each objective function evaluation made during a local search """

    def __init__(self):
        self.cost_sequences = list()        # A list of lists, were each list is a sequence of solution objective function values for one run of the algorith,
        self.best_cost_sequences = list()   # A list of lists, were each list is a sequence of the best-so-far solution objective function values for one run of the algorith,
        self.legends = list()               # The legend text to show for a run's data plot

    def initialise_new_run(self, label =""):
        """
        Call this when a new run of the algorithm is being started.
        Args: label: optional string to show in the legend for this sequence of costs
        """
        self.cost_sequences.append(list())
        self.best_cost_sequences.append(list())
        self.legends.append(label)
    
    def add_new_point(self,cost, best_cost = None):
        """
        Add a new objective function value to data to plot for the current run. 
        Args:   cost: objective function value
                best_cost (optional): if specified, is an algorithm-specific "best objective function" value at this step 
                                      if not specified, is set as the best objective function value so far in this run
        """
        if not best_cost:
            best_cost = min(self.best_cost_sequences[-1][-1],cost) if self.best_cost_sequences[-1] else cost
        self.cost_sequences[-1].append(cost)
        self.best_cost_sequences[-1].append(best_cost)
    
    def showplot(self,title : str = "Solution Quality vs Function Evaluation Count"):
        """ Plot the data using Plotly for collected sequence of runs. (This will open in a browser outside of a Jupyter notebook.) """
        count = 0
        fig = plotly.express.line(labels={'x': 'Iteration Count', 'y': 'Cost'},
                    title=title)
        for cost_sequence, best_cost_sequence, legend in zip(self.cost_sequences, self.best_cost_sequences, self.legends):
            fig.add_scatter(x=list(range(count,count+len(cost_sequence))), y=cost_sequence, mode='lines+markers', line=dict(color="#0000ff"), name=legend)
            fig.add_scatter(x=list(range(count,count+len(best_cost_sequence))), y=best_cost_sequence, mode='lines+markers', line=dict(color="#00ffff",width=4), marker=dict(size=5, color="LightSeaGreen"), opacity=0.5, name=legend)       
            count += len(cost_sequence) 
        fig.show()

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
            if i==j:
                continue
            else:
                cost += F[i, j] * D[x[i], x[j]]
    return cost

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

def nextdescent_to_local_minimum(x, F, D, plot_manager : PlotManager = None):
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
    if plot_manager: plot_manager.initialise_new_run("Next Descent")
    n = len(x)
    ij_comb = (-1,-1)

    # TODO FIX THE FOLLOWING CODE
    for p in range(n-1):
        for q in range(p+1, n):
            if (p,q) == ij_comb:
                # Finish if we see the same solution
                p = n
                break

            change_in_cost, x_test = compute_cost_change(x,F,D,p,q)
            
            if change_in_cost < 0:
                # If we get an improvement in cost, keep the solution
                x = x_test.copy()
                ij_comb = (p,q)
                cost += change_in_cost

            if plot_manager: plot_manager.add_new_point(cost+change_in_cost, cost)
    return x, cost

def do_random_restart_nextdescent(F, D):
    """
    Generate a random starting solution, and descend from it to a local minimum using nextdescent
    Returns the local minimum x and its cost c
    """
    n = np.shape(F)[0]   # Number of facilities is given by number of rows (or columns) in F
    x = np.random.permutation(n)  # Initial random permutation
    return nextdescent_to_local_minimum(x, F, D)    

def steepestdescent_to_local_minimum(x, F, D, plot_manager : PlotManager = None):
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
    if plot_manager: plot_manager.initialise_new_run("Steepest Descent")
    # TODO Fix this code
    n = len(x)
    while True:
        # Get all neighbours by finding possible swaps
        # Data structure is set up such as (cost, solution)
        neighbours = [(compute_cost_change(x, F, D, i, j)) 
                       for j in range(1,n)
                       for i in range (0,n-1)]

        # Get the minimum of the neighbours, test for local minimum
        best_neighbour = (cost, x)

        for neighbour in neighbours:
            if neighbour[0] < best_neighbour[0]:
                best_neighbour = neighbour
            if plot_manager : plot_manager.add_new_point(cost + neighbour[0], cost) # Plot
        
        # This line was just to verify that my similar looking minima were not repeated neighbours
        # Uncomment to verify for yourself
        # print(f'Lowest cost: {best_neighbour[0]}')

        if best_neighbour[0] >= 0:
            # Local minimum has been reached
            break
        else:
            # Add to new cost
            x = best_neighbour[1].copy()
            cost += best_neighbour[0]
    
    return x, cost

def do_random_restart_steepestdescent(F, D):
    """
    Generate a random starting solution, and descend from it to a local minimum using nextdescent
    Returns the local minimum x and its cost c
    """
    x = np.random.permutation(n)  # Initial random permutation
    return steepestdescent_to_local_minimum(x, F, D)

def randsomsearch(numsolutions, F, D, plot_manager : PlotManager = None):
    """Generate numsolutions random solutions, returning the best of these and its cost"""
    n = np.shape(F)[0]
    if plot_manager: plot_manager.initialise_new_run("Random Search")
    x_best = None
    cost_best = float('inf')
    for i in range(numsolutions):
        x = np.random.permutation(n)
        cost = compute_cost(x, F, D)
        if plot_manager: plot_manager.add_new_point(cost)
        if cost < cost_best:
            x_best = x
            cost_best = cost
    return x_best, cost_best

def test_algorithm(algorithm, max_run_time):
    """
    Repeatedly run the specified algortithm until max_run_time seconds have passed, collecting the run times 
    and solution objectie function values. THe solutions found are printed.
    Args:
        algorithm: one of do_random_restart_steepestdescent or do_random_restart_nextdescent
        max_run_time: How many seond to run for
    Returns:
        best_x best solution found 
        best_cost: objective function value of best_x
        cost_sequence: the sequence of objective function values found by each run of the algorithm
        best_cost_sequence: the sequence of best-so-far objective function values  
        time_sequence: the sequence of times (0 being the start of the test) when each objective function value was found
    """
    best_cost = float('inf')
    best_x = None
    cost_sequence = list()
    best_cost_sequence = list()
    time_sequence = list()
    start_time = timeit.default_timer()
    end_time = start_time + max_run_time
    i=1
    while timeit.default_timer() < end_time:
        x, cost = algorithm(F, D)
        if cost < best_cost:
            best_cost = cost
            best_x = x
        cost_sequence.append(cost)
        best_cost_sequence.append(best_cost)
        time_sequence.append(timeit.default_timer() - start_time)
        print(f"{i:3} f(x)={cost}, x={x}")
        i=i+1
    return best_x, best_cost, cost_sequence, best_cost_sequence, time_sequence

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

def do_tabusearch(F, D):
    """
    Uses x = np.array([5,0,4,3,2,1]), then tries to find a better solution repeatably until f(x) is minimised.
    This function uses the tabu search method
    """
    x = np.array([5,0,4,3,2,1])
    x_new, cost = tabusearch(x, F, D)
    return x_new, cost

def do_tabusearch2(F, D):
    """
    Generates a random solution, x, then tries to find a better solution repeatably until f(x) is minimised.
    This function uses the tabu search method
    """
    # First, initialise some random solution
    x = np.random.permutation(n)  # Initial random permutation
    x_new, cost = tabusearch(x, F, D)
    return x_new, cost

def tabusearch(x, F, D, plot_manager = None):
    """
    Apply the tabu search method over 30 iterations
    Args:
        x             : Current solution (x[i] is the location index for facility i)
        F (np.ndarray): Flow matrix.
        D (np.ndarray): Distance matrix.
        plot_manager  : An optional plot manager to collect data for plotting
    Returns:
        x (list): Best solution found.
        cost (float): Objective function value of solution x
    """
    n = len(x)
    cost = compute_cost(x, F, D)

    # Initialise the history
    # Use a dictionary for every possible change and whether it is banned or not
    H = {}
    for a in range(0, n-1):
        for b in range(a+1, n):
            H[f'{a},{b}'] = 'notban'
            H[f'{a},{b}'] = 'notban'

    # Initialise temporary variables
    x_new = x.copy()
    cost_new = cost

    # Use a heap for tracking which moves to unban
    heap = []

    # Init so never goes off first iteration
    heap_size = np.inf

    # New search
    if plot_manager: plot_manager.initialise_new_run("Tabu search")

    for k in range(30):
        # Unban swaps at bottom of heap
        if k > heap_size:
            swap = heap.pop(0)
            H[swap] = 'notban'
            H[swap[::-1]] = 'notban'
        
        # Calculate neighbourhood
        nhood = [(compute_cost_change(x_new, F, D, i, j), f'{i},{j}') 
                    for i in range (0,n-1) 
                    for j in range(i+1,n)
                    if H[f'{i},{j}'] == 'notban'] # Avoid banned swaps
        
        # Update heap size on neighbourhood size:
        if k == 0:
            heap_size = np.round(0.5 * len(nhood))
        
        # Define best solution
        best_sol = ((np.inf, x), '-1,-1')

        for nbr in nhood:
            # Plot point
            if plot_manager: plot_manager.add_new_point(cost + nbr[0][0], cost_new)

            # Get best solution
            if nbr[0][0] < best_sol[0][0]:
                # Update best solution
                best_sol = nbr

        # Update cost + solution
        cost_new = cost + best_sol[0][0]
        x_new = best_sol[0][1].copy()

        # Update global best solution
        if cost_new < cost:
            cost = cost_new
            x = x_new.copy()

        # Ban best swap
        heap.append(best_sol[1])
        H[best_sol[1]] = 'ban'
        H[best_sol[1][::-1]] = 'ban'

    return x, cost

if __name__ == "__main__":
    # Example flow and distance matrices for testing & plotting
    n,F,D = create_small_instance()

    # We will do detailed plotting of the runs
    plot_manager = PlotManager()

    # Commented out as random search is not needed for hand-in
    # Generate 100 random solutions and add them to the plot
    random.seed(12345)
    x, cost = randsomsearch(100, F, D, plot_manager)

    # Generate multiple local optima from different starting solutions using Next Descent, adding each search to the plot
    x=[0,1,2,3,4,5]
    x, cost = nextdescent_to_local_minimum(x, F, D, plot_manager)
    x=[3,2,1,0,4,5]
    x, cost = nextdescent_to_local_minimum(x, F, D, plot_manager)
    x=[4,3,2,1,0,5]
    x, cost = nextdescent_to_local_minimum(x, F, D, plot_manager)
    x=[5,0,4,3,2,1]
    x, cost = nextdescent_to_local_minimum(x, F, D, plot_manager)
    x=[5,1,4,3,2,4]
    x, cost = nextdescent_to_local_minimum(x, F, D, plot_manager)

    # Plot results
    # plot_manager.showplot('Next descent algorithm')

    # Repeat with steepest descent ##################################################
    # Second plotter for the steepest descent routine
    # plot_manager = PlotManager()

    x=[0,1,2,3,4,5]
    x, cost = steepestdescent_to_local_minimum(x, F, D, plot_manager)
    x=[3,2,1,0,4,5]
    x, cost = steepestdescent_to_local_minimum(x, F, D, plot_manager)
    x=[4,3,2,1,0,5]
    x, cost = steepestdescent_to_local_minimum(x, F, D, plot_manager)
    x=[5,0,4,3,2,1]
    x, cost = steepestdescent_to_local_minimum(x, F, D, plot_manager)

    # Plot results of steepest descent
    # plot_manager.showplot('Steepest descent')

    # Plot these random and local search results
    plot_manager.showplot("Random, Next & Steepest Descent : Solution Quality vs Function Evaluation Count.")

    # Tabu search
    plot_manager2 = PlotManager()
    # Generate 100 random solutions and add them to the plot
    random.seed(12345)

    # x, cost = randsomsearch(100, F, D, plot_manager2)
    x, cost = tabusearch([5,0,4,3,2,1], F, D, plot_manager2) 

    # Plot search results
    plot_manager2.showplot('Tabu search')   

    # Create a larger instance
    n = 30  # Number of facilities
    random.seed(12345)
    F = np.random.randint(1, 10, size=(n, n))  # Random flow matrix
    D = np.random.randint(1, 10, size=(n, n))  # Random distance matrix
    for i in range(n):
        F[i][i]=0
        D[i][i]=0

    # Test Next Descent and Steepest Descent by plotting solution quality vs time
    runtime = 30 # seconds
    print("Testing Next Descent")
    random.seed(12345)
    best_x, best_cost, nd_cost_sequence, nd_best_cost_sequence, nd_time_sequence = test_algorithm(do_random_restart_nextdescent,runtime)
    print("Testing Steepest Descent")
    random.seed(12345)
    best_x, best_cost, sd_cost_sequence, sd_best_cost_sequence, sd_time_sequence = test_algorithm(do_random_restart_steepestdescent,runtime)
    print('Testing Tabu Search')
    random.seed(12345)
    best_x, best_cost, ts_cost_sequence, ts_best_cost_sequence, ts_time_sequence = test_algorithm(do_tabusearch2,runtime)

    # Plot the results
    fig = plotly.express.line(labels={'x': 'Time (s)', 'y': 'Cost'}, title="Performance vs time")
    fig.add_scatter(x=nd_time_sequence, y=nd_cost_sequence, mode='markers', line=dict(color="#ff0000"),name="Next Descent")
    fig.add_scatter(x=nd_time_sequence, y=nd_best_cost_sequence, mode='lines+markers', line=dict(color="#ffff00",width=4), marker=dict(size=5, color="LightSeaGreen"), opacity=0.5,name="Next Descent")       
    fig.add_scatter(x=sd_time_sequence, y=sd_cost_sequence, mode='markers', line=dict(color="#0000ff"),name="Steepest Descent")
    fig.add_scatter(x=sd_time_sequence, y=sd_best_cost_sequence, mode='lines+markers', line=dict(color="#00ffff",width=4), marker=dict(size=5, color="LightSeaGreen"), opacity=0.5,name="Steepest Descent")       
    fig.add_scatter(x=ts_time_sequence, y=ts_cost_sequence, mode='markers', line=dict(color="#00ff00"),name="Tabu Search")
    fig.add_scatter(x=ts_time_sequence, y=ts_best_cost_sequence, mode='lines+markers', line=dict(color="#333333",width=4), marker=dict(size=5, color="LightSeaGreen"), opacity=0.5,name="Tabu Search")       
    fig.show()

