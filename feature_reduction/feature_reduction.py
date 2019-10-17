import numpy as np
import matplotlib.pyplot as plt

def plot_scatters(scatter_list, scatter_names, label = "2D Gaussian Scatter Plots", arb_line = False, proj_x = False, proj_y = False, pca = False):
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
    plt.title(label)
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
    scatter_list.append(generate_gaussian_scatter(mean = [-5,5], cov = [[1, 1.5], [1.5, 5]], num_samp = 1000))
    scatter_list.append(generate_gaussian_scatter(mean = [5,3], cov = [[1, 1.5], [1.5, 5]], num_samp = 1000))
    
    scatter_names = ["2D-Gaussian Blue", "2D-Gaussian Orange"]
    
    fig, ax = plot_scatters(scatter_list, scatter_names)
    fig, ax = plot_scatters(scatter_list, scatter_names, label = "2D Gaussian Scatter Plots with Arbitrary Threshold", arb_line = True)
    fig, ax = plot_scatters(scatter_list, scatter_names, label = "2D Gaussian Scatter Plots Projected on X-axis", proj_x = True)
    fig, ax = plot_scatters(scatter_list, scatter_names, label = "2D Gaussian Scatter Plots Projected on Y-axis", proj_y = True)

def curse_of_dimensionality():
    num_samples = 100
    dim = range(1, 500)
    
    r_mean = []
    r_sigma = []
    
    for d in dim:
        # Generate m-dimensional gaussian points and calculate radius and sigma
        normal_deviates = np.random.normal(size=(d, num_samples))
        radius = np.sqrt((normal_deviates**2).sum(axis=0))
        sigma = np.std(radius)
        
        r_mean.append(np.mean(radius))
        r_sigma.append(sigma)
    
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(dim, r_mean)
    ax.set(xlabel='Dimensions', ylabel='Radius mean', title='Radius vs dimensions - 100 samples')
    plt.show
    
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(dim,r_sigma)
    ax.set(xlabel='Dimensions', ylabel='Sigma', title='Sigma vs. dimensions - 100 samples')
    plt.show()
    
def pca(scatter_list):
    scatter_list_pca = np.concatenate((scatter_list[0], scatter_list[1]))
    
    mean = np.mean(scatter_list_pca.T, axis=1)      
    center = scatter_list_pca - mean
    cov = np.cov(center.T)
    eig_val, eig_vec = np.linalg.eig(cov)
    
    print(eig_val)
    print(center.shape)
    
    PC = eig_vec.T[0]
    
    proj1 = np.dot(center[:1000], PC.T)
    proj2 = np.dot(center[1000:], PC.T)
    #proj = (eig_vec.T.dot(center.T))
    
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    fig, ax = plt.subplots(figsize=(7, 7))
    
    for s in range(len(scatter_list)):
        ax.scatter(scatter_list[s][:,0], scatter_list[s][:,1], c= colors[s], s=10.0, alpha=0.3)
    
    ax.scatter(np.zeros(len(proj1)), proj1, c = colors[0], s=10.0, alpha=0.3)
    ax.scatter(proj2, np.zeros(len(proj2)), c = colors[1], s=10.0, alpha=0.3)
    
    # Create x and y axis lines    
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    
    plt.show()
    
def principle_component_analysis():
    scatter_list_1 = generate_gaussian_scatter(mean = [-5,5], cov = [[5, 1.5], [1.5, 0.5]], num_samp = 1000)
    scatter_list_2 = generate_gaussian_scatter(mean = [5,3], cov = [[5, 1.5], [1.5, 0.5]], num_samp = 1000)
    
    scatter_list_pca = np.concatenate((scatter_list_1, scatter_list_2))
    scatter_list = [scatter_list_1, scatter_list_2]
    
    
    scatter_names = ["2D-Gaussian Blue", "2D-Gaussian Orange"]
    
    
    
    pca(scatter_list)

def main():
    if False: projection_example()
    if False: curse_of_dimensionality()
    if True: principle_component_analysis()

if __name__ == '__main__':
    main()
