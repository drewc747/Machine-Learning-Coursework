import numpy as np
import matplotlib.pyplot as plt

def plot_scatters(scatter_list, scatter_names, arb_line = False, proj_x = False, proj_y = False):
    '''
    Method to plot a multiple scatter plots from a list
        scatter_list: List of np arrays, each array should contain scatter data (x, y)
        scatter_names: List of names for each scatter list to be plotted
    '''
    
    # Get list of default colors
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    fig, ax = plt.subplots(figsize=(7, 7))
    for s in range(len(scatter_list)):
        ax.scatter(scatter_list[s][:,0], scatter_list[s][:,1], c= colors[s], s=10.0, alpha=0.3, label=scatter_names[s])
        
        if proj_x:
            ax.scatter(scatter_list[s][:,0], np.zeros(scatter_list[s][:,1].shape), c= colors[s], s=15.0, label= scatter_names[s] + " X Proj", marker = "+")
        if proj_y:
            ax.scatter(np.zeros(scatter_list[s][:,0].shape), scatter_list[s][:,1], c= colors[s], s=15.0, alpha = 0.15, label=scatter_names[s] + " Y Proj", marker = "+")
    
    # Create x and y axis lines    
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    
    if arb_line:
       x = np.linspace(-4, 6, 10)
       ax.plot(x, 1.5*x + 2, linestyle='-', color = 'r', label="Arbitrary Threshold Line")
    
    ax.legend(loc=0)
    plt.show()
    
    return fig, ax
    
def generate_gaussian_scatter(mean, cov, num_samp):
    '''
    Method to generate a gaussian scatter
        mean: List of x and y values for mean
        cov: List of lists representing the covariance matrix, 2x2 matrix
        num_samp: Number of samples
    '''
    np.random.seed(1024)
    gaussian_scatter = np.random.multivariate_normal(mean, cov, num_samp)
    return gaussian_scatter

def projection_example():
    scatter_list = []
    scatter_list.append(generate_gaussian_scatter(mean = [-5,5], cov = [[1, 0], [0, 5]], num_samp = 1000))
    scatter_list.append(generate_gaussian_scatter(mean = [5,3], cov = [[1, 1.5], [1.5, 5]], num_samp = 1000))
    
    scatter_names = ["2D-Gaussian Blue", "2D-Gaussian Orange"]
    
    fig, ax = plot_scatters(scatter_list, scatter_names, proj_y = True)
    
def main():
    projection_example()

if __name__ == '__main__':
    main()
