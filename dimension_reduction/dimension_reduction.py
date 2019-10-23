import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn import decomposition
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

def plot_scatters(scatter_list, scatter_names, title = "2D Gaussian Scatter Plots", arb_line = False, proj_x = False, proj_y = False, pca = False, pc = None, proj_pca = False):
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
        if proj_pca:
            if s < len(scatter_list)/2:
                ax.scatter(scatter_list[s][:,0], scatter_list[s][:,1], c= colors[s], s=10.0, alpha=0.3, label=scatter_names[s])
            else:
                ax.scatter(scatter_list[s][:,0], scatter_list[s][:,1], c= colors[s - int(len(scatter_list)/2)], s=15.0, alpha = 0.3, marker = "+", label = scatter_names[s])
        else:
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
        x = np.linspace(-10, 6, 4)
        ax.plot(x, 1.5*x + 6, linestyle='-', color = 'r', label="Arbitrary Threshold Line")
    
    if pca and pc is not None:
        ax.plot([-10*pc[0][0], 10*pc[0][0]],[-10*pc[1][0], 10*pc[1][0]], linestyle='--', color = colors[2], label="1st eigenvector direction")
        ax.plot([-10*pc[0][1], 10*pc[0][1]],[-10*pc[1][1], 10*pc[1][1]], linestyle='--', color = colors[3], label="2nd eigenvector direction")   
            
    ax.legend(loc=0)
    plt.title(title)
    plt.show()
    
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
    scatter_list.append(generate_gaussian_scatter(mean = [-7,5], cov = [[1, 1.5], [1.5, 5]], num_samp = 1000))
    scatter_list.append(generate_gaussian_scatter(mean = [5,3], cov = [[5, 1.5], [1.5, 0.5]], num_samp = 1000))
    
    scatter_names = ["2D-Gaussian Blue", "2D-Gaussian Orange"]
    
    plot_scatters(scatter_list, scatter_names)
    plot_scatters(scatter_list, scatter_names, title = "2D Gaussian Scatter Plots with Arbitrary Threshold", arb_line = True)
    plot_scatters(scatter_list, scatter_names, title = "2D Gaussian Scatter Plots Projected on X-axis", proj_x = True)
    plot_scatters(scatter_list, scatter_names, title = "2D Gaussian Scatter Plots Projected on Y-axis", proj_y = True)

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
    plt.show()
    
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(dim,r_sigma)
    ax.set(xlabel='Dimensions', ylabel='Sigma', title='Sigma vs. dimensions - 100 samples')
    plt.show()

def normalize(X):
    mean = np.mean(X, axis=0)
    return X - mean

def get_eigs(X):
    cov = np.cov(X.T)
    _, eig_val, eig_vec = np.linalg.svd(cov)
    return eig_val, eig_vec

def pca_2d_to_1d(scatter_list, scatter_names, num_samples):
    #plot original data
    plot_scatters(scatter_list, scatter_names)
    
    # Normalize data and get principle components
    center = normalize(np.concatenate(scatter_list))
    eig_val, eig_vec = get_eigs(center)
    print(eig_val)
    print(eig_vec)
    pc = eig_vec.T
    
    # Plot normalized data with eigenvectors
    norm = np.split(center, len(scatter_list))
    norm_names = ["Normalized " + s for s in scatter_names] 
    plot_scatters(norm, norm_names, pca = True, pc = pc, title = "Normalized Data with PCA axes")

    # Project rotated data in original space
    x_rot = np.dot(center, pc.T)
    proj = np.zeros(x_rot.shape)
    proj[:,0] = x_rot[:,0]
    proj = np.dot(np.linalg.inv(pc), proj.T).T
    
    # Plot normalized data with projected data to 1st eigenvector
    proj_list = np.split(proj, len(scatter_list))
    proj_names = [s + " PCA Projection onto 1st Eigenvector" for s in norm_names]
    plot_scatters(scatter_list + proj_list, norm_names + proj_names, title = "Normalized Data and Data projected onto 1st Eigenvector", proj_pca = True)
    
    #plot histogram
    proj_points = np.split(x_rot[:,0], len(scatter_list))
    hist_names = ["PCA Projections on 1st Eigenvector " + s for s in scatter_names]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set(xlabel='X* - Projected Axis', ylabel='Count', title='Histogram of data points on projected axis')
    ax.hist(proj_points, bins=50, label = hist_names)
    ax.legend(loc=0)
    plt.show()
    
def principle_component_analysis():
    
    # Example 1 - Original example
    scatter_list = []
    num_samples = 1000
    scatter_list.append(generate_gaussian_scatter(mean = [-7,5], cov = [[1, 1.5], [1.5, 5]], num_samp = num_samples))
    scatter_list.append(generate_gaussian_scatter(mean = [5,3], cov = [[5, 1.5], [1.5, 0.5]], num_samp = num_samples))
    
    scatter_names = ["2D-Gaussian Blue", "2D-Gaussian Orange"]
       
    pca_2d_to_1d(scatter_list, scatter_names, num_samples)

    # Example 2 - 3 Class PCA
    scatter_list = []
    num_samples = 1000
    scatter_list.append(generate_gaussian_scatter(mean = [-0.5,2], cov = [[3, 1.5], [1.5, 1]], num_samp = num_samples))
    scatter_list.append(generate_gaussian_scatter(mean = [3.5,-3.5], cov = [[3, 1.5], [1.5, 1]], num_samp = num_samples))
    scatter_list.append(generate_gaussian_scatter(mean = [-7,5], cov = [[1, 1.5], [1.5, 5]], num_samp = num_samples))
    
    scatter_names = ["2D-Gaussian Blue", "2D-Gaussian Orange", "2-D Gaussian Green"]
    
    pca_2d_to_1d(scatter_list, scatter_names, num_samples)
    
    # Example 3 - PCA Fails
    scatter_list = []
    num_samples = 1000
    scatter_list.append(generate_gaussian_scatter(mean = [0,1], cov = [[5, 1.5], [1.5, 0.5]], num_samp = num_samples))
    scatter_list.append(generate_gaussian_scatter(mean = [0,-1], cov = [[5, 1.5], [1.5, 0.5]], num_samp = num_samples))
    
    scatter_names = ["2D-Gaussian Blue", "2D-Gaussian Orange"]
       
    pca_2d_to_1d(scatter_list, scatter_names, num_samples)

def get_age(rings):
    if 1 <= rings < 8.5:
        return 0
    elif 8.5 <= rings < 11.5:
        return 1
    else:
        return 2

def linear_discriminant_analysis():
    # Load the dataset
    #dataset = 'iris'
    dataset = 'abalone'
    df = pd.read_csv(f'./data/{dataset}.csv')
    
    if dataset == 'abalone':
        class_col = 'Age'
    elif dataset == 'iris':
        class_col = 'Species'
    
    # Look over data
    print(df.sample(10))
    print(df.describe(include = 'all'))
    
    # Clean data
    if dataset == 'abalone':
        print(df[df.Height == 0]) # 0 Height is suspicious
        df = df[df.Height != 0] # Remove 0 height
    print(df.isna().sum()) # No NaNs
    
    if dataset == 'abalone':
        # Change Sex to boolean columns
        #new_cols = pd.get_dummies(df.Sex)
        #df[new_cols.columns] = new_cols
        df.drop(columns = ['Sex'], inplace = True)
        print(df.describe(include = 'all'))
    
        # Change rings into 3 classes
        df[class_col] = df['Rings'].map(get_age)
        df.drop(columns = ['Rings'], inplace = True)
        print(df.sample(10))
        print(df.describe(include = 'all'))
    
    sns.pairplot(df, hue = class_col)
    plt.show()
    
    corr = df[df.columns].corr()
    sns.heatmap(corr, annot=True)
    plt.show()
    #df.hist(column = class_col, bins = 3)
    #plt.show()
    
    # Calculate in class means
    class_means = df.groupby(class_col).mean().transpose()
    print(class_means)
    # Calculate within class scatteer matrix
    num_features = class_means.shape[0]
    sm_within = np.zeros((num_features, num_features))
    for c, rows in df.groupby(class_col):
        rows = rows.drop([class_col], axis=1)
        s = np.zeros((num_features, num_features))
        for idx, row in rows.iterrows():
            x = row.values.reshape(num_features, 1)
            mc = class_means[c].values.reshape(num_features, 1)
            
            s += (x - mc).dot((x - mc).T)
            sm_within += s
            
    # Calculate between class scatter matrix
    feature_means = df.drop(columns = [class_col]).mean()
    
    sm_between = np.zeros((num_features, num_features))
    for c in class_means:
        n = len(df.loc[df[class_col] == c].index)
        mc = class_means[c].values.reshape(num_features,1)
        m = feature_means.values.reshape(num_features,1)
        
        sm_between += n * (mc - m).dot((mc - m).T)
    
    # Calculate eigen_values and vectors
    eig_val, eig_vec = np.linalg.eig(np.linalg.inv(sm_within).dot(sm_between))
    
    pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range (len(eig_val))]
    pairs = sorted(pairs, key=lambda x: x[0], reverse=True)
    
    for pair in pairs:
        print(pair[0])
    
    # Matrix with first two eigenvectors
    w_matrix = np.hstack((pairs[0][1].reshape(num_features,1), pairs[1][1].reshape(num_features,1))).real
    
    x_lda = np.array(df.drop([class_col], axis=1).dot(w_matrix))
    le = LabelEncoder()
    y = le.fit_transform(df[class_col])
    
    # Create colormap
    colors = plt.get_cmap("tab10")
    colors = truncate_colormap(colors, 0.0, 0.2)
    
    # Plot results
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x_lda[:,0], x_lda[:,1], c=y, cmap= plt.get_cmap("tab10")[0:2], alpha =0.7)
    
    # Create x and y axis lines    
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    
    plt.xlabel('LD1')
    plt.ylabel('LD2')
    
    plt.show()
    
    # PCA for comparison
    X = np.array(df.drop([class_col], axis=1))
    print(X)
    Y = np.array(df[class_col])
    
    pca = decomposition.PCA(n_components = 2)
    pca.fit(X)
    X = pca.transform(X)
    
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(X[:, 0], X[:, 1], c = Y, cmap = 'tab10', alpha =0.7, label = Y)
    
    # Create x and y axis lines    
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    
    ax.legend(loc=0)
    plt.title(f'PCA on {dataset} dataset')
    plt.show()
    
    
def main():
    if False: projection_example()
    if False: curse_of_dimensionality()
    if False: principle_component_analysis()
    if True: linear_discriminant_analysis()

if __name__ == '__main__':
    main()
