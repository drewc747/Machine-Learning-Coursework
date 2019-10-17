import numpy as np
import matplotlib.pyplot as plt
'''
mean_blue = [-5, 5]
mean_red = [5, 3]

cov = [[1, 1.5], [1.5, 5]]

np.random.seed(1024)
X_blue = np.random.multivariate_normal(mean_blue, cov, 1000)
X_red = np.random.multivariate_normal(mean_red, cov, 1000)

fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(X_blue[:,0], X_blue[:,1], c='b', s=10.0, alpha=0.3, label = "2D-Gaussian Blue")
ax.scatter(X_red[:,0], X_red[:,1], c='r', s=10.0, alpha=0.3, label = "2D-Gaussian Red")
ax.grid()
ax.legend(loc=0)
ax.set_xlim([-15, 15])
ax.set_y_lim([-10,20])
plt.show()
'''

def plot_scatters(scatter_list, scatter_names):
    fig, ax = plt.subplots(figsize=(7, 7))
    
    for s in range(len(scatter_list)):
        ax.scatter(scatter_list[s][:,0], scatter_list[s][:,1], s=10.0, alpha=0.3, label=scatter_names[s])
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    ax.legend(loc=0)
    plt.show()
    
def generate_gaussian_scatter(mean, cov, num_samp):
    np.random.seed(1024)
    gaussian_scatter = np.random.multivariate_normal(mean, cov, num_samp)
    return gaussian_scatter

def projection_example():
    scatter_list = []
    scatter_list.append(generate_gaussian_scatter(mean = [-5,5], cov = [[1, 1.5], [1.5, 5]], num_samp = 1000))
    scatter_list.append(generate_gaussian_scatter(mean = [5,3], cov = [[1, 1.5], [1.5, 5]], num_samp = 1000))
    
    scatter_names = ["2D-Gaussian Blue", "2D-Gaussian Set Orange"]
    
    plot_scatters(scatter_list, scatter_names)
    
def main():
    projection_example()

if __name__ == '__main__':
    main()
