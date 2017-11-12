import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from sklearn import metrics
from sklearn.cluster import DBSCAN
import treelib
import pyclust
from sklearn.cluster import hierarchical
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
import csv
import math
from random import randint
from scipy import stats
from statsmodels.stats.weightstats import ztest
from statsmodels.stats.weightstats import ttest_ind
from scipy.stats import chisquare
from CHAID import Tree

# get attribute

# data = pd.read_csv('attributes_05.15.2017.csv')

def preprocessing(filename):
    data = pd.read_csv(filename)
    if len(data[data['WithAttributes?']!= 'Yes']) == 0:
        data.drop('WithAttributes?',axis =1,inplace=True)
    for item in ['Response', 'INJ_TYPE', 'EdDispGroup']:
        for i in range(0, len(data.columns)):
            if data.columns[i] == item:
                loc = i
        ls = data[item].unique()
        for i in range(0, len(ls)):
            data.insert(loc+i,ls[i],pd.get_dummies(data[item])[ls[i]])
        data.drop(item,axis =1,inplace = True)

#####change non-critical admission & critical admission positions because the copofrand group
    non_critical = data['Non-critical admission']
    data.drop('Non-critical admission',axis =1,inplace =True)
    for i in range(0,len(data.columns)):
        if data.columns[i] == 'Critical admission ':
            loc = i
    data.insert(loc,'Non-critical admission',non_critical)

    attributes =[]
    for index in range(len(data)):
        attributes.append(data.drop('id',axis=1,inplace=False).iloc[index])
    # attributes = np.array(attributes)
    # dimension_reduced = PCA(n_components=2).fit_transform(attributes)
    return attributes,data

def get_weighted_attributes(attributes,weight):
    attributes= np.array(attributes,dtype='float64')
    maxAttr = 0
    for i in range(len(attributes[0])):
        maxAttr = float(max(attributes[:,i]))
        for j in range(len(attributes)):
            attributes[j][i] = float(float(attributes[j][i]) / maxAttr)
        maxAttr = 0
    weighted_attributes = np.array(attributes*weight,dtype ='float64')
    return weighted_attributes


def elbow_method(dimension_reduced):
    distortions = []
    K = range(2,10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(dimension_reduced)
        distortions.append(sum(np.min(cdist(dimension_reduced, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / dimension_reduced.shape[0])

    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    # plt.show()
    plt.savefig('The Elbow Method showing the optimal k')
    slope_1 =[]
    for i in range(1,8):
        slope_1.append(distortions[i-1]-distortions[i]) 

    slope_1.index(max(slope_1))
    slope_2 =[]
    for i in range(1,len(slope_1)):
        slope_2.append(distortions[i-1]-distortions[i]) 
    k = slope_2.index(max(slope_2))+4
    return k
def highly_weighted_attributes_name(data,weight):
    plt.hist(weight)
    plt.xlabel('Weights')
    plt.ylabel('Amounts')
    plt.title('The Distribution of Weights')
    index = []
    for i in range(len(weight)):
        if weight[i] > 15:
            index.append(i)
    
    index_column = []
    for item in index:
        index_column.append(data.columns[item+1])
    print index_column
    
    return index , index_column

def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    # rotate theta such that the first axis is at the top
    theta += np.pi/2

    def draw_poly_patch(self):
        verts = unit_poly_verts(theta)
        return plt.Polygon(verts, closed=True, edgecolor='k')

    def draw_circle_patch(self):
        # unit circle centered on (0.5, 0.5)
        return plt.Circle((0.5, 0.5), 0.5)

    patch_dict = {'polygon': draw_poly_patch, 'circle': draw_circle_patch}
    if frame not in patch_dict:
        raise ValueError('unknown value for `frame`: %s' % frame)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1
        # define draw_frame method
        draw_patch = patch_dict[frame]

        def fill(self, *args, **kwargs):
            """Override fill so that line is closed by default"""
            closed = kwargs.pop('closed', True)
            return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super(RadarAxes, self).plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            return self.draw_patch()

        def _gen_axes_spines(self):
            if frame == 'circle':
                return PolarAxes._gen_axes_spines(self)
            # The following is a hack to get the spines (i.e. the axes frame)
            # to draw correctly for a polygon frame.

            # spine_type must be 'left', 'right', 'top', 'bottom', or `circle`.
            spine_type = 'circle'
            verts = unit_poly_verts(theta)
            # close off polygon by repeating first vertex
            verts.append(verts[0])
            path = Path(verts)

            spine = Spine(self, spine_type, path)
            spine.set_transform(self.transAxes)
            return {'polar': spine}

    register_projection(RadarAxes)
    return theta


def unit_poly_verts(theta):
    """Return vertices of polygon for subplot axes.

    This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
    """
    x0, y0, r = [0.5] * 3
    verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
    return verts

def draw_radar(figdata,N,name,grids):
    theta = radar_factory(N, frame='polygon')
    spoke_labels = figdata.pop(0)

    fig, axes = plt.subplots(figsize=(24, 24), nrows=2, ncols=2,subplot_kw=dict(projection='radar'))

    colors = ['b', 'r', 'g', 'm', 'y']
# Plot the four cases from the example data on separate axes
    for ax, (title, case_data) in zip(axes.flatten(), figdata):
        ax.set_rgrids(grids)
        ax.set_title(title, weight='bold', size='x-large', position=(0.5, 1.1),horizontalalignment='center', 
                     verticalalignment='center')
        for d, color in zip(case_data, colors):
            ax.plot(theta, d, color=color)
            ax.fill(theta, d, facecolor=color, alpha=0.25)
        ax.set_varlabels(spoke_labels)
    fig.text(0.5, 0.965, name,
             horizontalalignment='center', color='black', weight='bold',
             size='xx-large')

    # plt.show()
    plt.savefig(name)

def Z_test(data):
    ztest_list =[]
    index = []
    ind = 0
    for item in data.columns[1:len(data.columns)-1]:
        if(ztest(data[data['label']==0][item],data[data['label']!=0][item])[1])<0.05:
                ztest_list.append(item)
                index.append(ind)
        ind += 1       
    

    return index, ztest_list

def T_test(data):
    ttest_list =[]
    index = []
    ind = 0

    for item in data.columns[1:len(data.columns)-1]:
        if(stats.ttest_ind(data[data['label']==0][item],data[data['label']!=0][item])[1])<0.05:
            ttest_list.append(item)
            index.append(ind)
        ind += 1

    return index, ttest_list

def Chi_Square_test(data):
    chisq_list =[]
    index = []
    ind = 0
    for item in data.columns[1:len(data.columns)-1]:
        if(chisquare(data[item])[1])<0.05:
            chisq_list.append(item)
            index.append(ind)
        ind += 1
    return index, chisq_list

def get_figdata(index,index_list,kmeans):
    group0 = []
    group1 = []
    group2 = []
    group3 = []
    for item in index:
        group0.append(kmeans.cluster_centers_[0][item])
        group1.append(kmeans.cluster_centers_[1][item])
        group2.append(kmeans.cluster_centers_[2][item])
        group3.append(kmeans.cluster_centers_[3][item])    
    
    figdata = [index_list,('group_0', [group0]),('group_1', [group1]),('group_2', [group2]),('group_3', [group3])]
    N = len(index)
    # draw_radar(figdata,N,name='ttest')
    return figdata

if __name__ == "__main__":
    
    filename = 'attributes_05.15.2017.csv'
    attributes,data = preprocessing(filename)
    dimension_reduced = PCA(n_components=2).fit_transform(attributes)
    k = elbow_method(dimension_reduced)					
    # best k =4
    # K-means
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(dimension_reduced)
    plt.scatter(dimension_reduced[:,0],dimension_reduced[:,1],c=kmeans.labels_,cmap='rainbow')
    plt.title('K-means')
    plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],c = 'black')
    plt.savefig('K-means clustering')
    print "kmeans-cluster centers:"
    print kmeans.cluster_centers_
    
    # K-medoid
    kmd = pyclust.KMedoids(n_clusters=4, n_trials=50)
    kmd.fit(dimension_reduced)
    plt.scatter(dimension_reduced[:,0],dimension_reduced[:,1],c=kmd.labels_,cmap ='rainbow')
    plt.title('K-medoid')
    plt.scatter(kmd.centers_[:,0],kmd.centers_[:,1],c = 'black')
    plt.savefig('K-medoid clustering')
    print "kmedoid-cluster centers:"
    print kmd.centers_
    
    #Density-based spatial clustering of applications with noise 
    db = DBSCAN(eps=0.6, min_samples=6).fit(dimension_reduced)
    #Scikit learn uses -1 to for NOISE, and starts cluster labeling at 0. 
    for i in range(0, len(db.labels_)):
        if not db.labels_[i] == -1:
            db.labels_[i] += 1
    plt.scatter(dimension_reduced[:,0],dimension_reduced[:,1],c=db.labels_,cmap='rainbow')
    plt.title('DBSCAN')
    plt.savefig('DBSCAN clustering')

    #Hierarchical: Agglomerative clustering
    link ='ward'
    ac = AgglomerativeClustering(linkage=link, n_clusters=4)
    ac.fit(dimension_reduced)
    # plt.scatter(dimension_reduced[:,0],dimension_reduced[:,1],c=ac.labels_,cmap='rainbow')
    # create linkage matrix
    Z = hierarchy.linkage(dimension_reduced, 'ward')
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    hierarchy.dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
    )
    # plt.show()
    plt.savefig('Hierarchical clustering')

    #####After weighted
    weight = [2, 2, 2, 3, 23, 4, 37, 31, 
    7, 35, 3, 8, 13, 15, 4, 19, 2, 21, 2, 3, 25, 19, 1, 4, 4, 21]
    weighted_attributes = get_weighted_attributes(attributes,weight)
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(weighted_attributes)
    
    
    #draw radar chart
    index,index_column = highly_weighted_attributes_name(data,weight)
    figdata = get_figdata(index,index_column,kmeans)
    N = len(index)
    draw_radar(figdata,N,'Critical attributes of a patient according to weights',[5,10,15,20,25,30,35])

    #kmeans without dimension reduced attributes
    kms = KMeans(n_clusters=4)
    kms.fit(attributes)
    data['label'] = kms.labels_

    #test
    ##### z-test
    index, ztest_list= Z_test(data)
    figdata = get_figdata(index,ztest_list,kms)
    draw_radar(figdata,len(index),'Z test',[0.5, 1, 1.5, 2, 2.5,3,3.5])

    ##### t-test
    index, ttest_list = T_test(data)
    figdata = get_figdata(index,ttest_list,kms)
    draw_radar(figdata,len(index),'T test',[0.5, 1, 1.5, 2, 2.5,3,3.5])


    ##### Chi-square test
    index, chisq_list = Chi_Square_test(data)
    figdata = get_figdata(index,chisq_list,kms)
    draw_radar(figdata,len(index),'Chi-Square test',[0.5, 1, 1.5, 2, 2.5,3,3.5])

    ####Chaid tree
    independent_variable_columns = data.columns[1:len(data.columns)-1]
    d_variable = data.columns[-1]
    dic ={}
    for item in independent_variable_columns[1:len(data.columns)-1]:
        dic.update({item:'nominal'})

    tr = Tree.from_pandas_df(data.drop(data.columns[0],axis= 1), dic,d_variable, max_depth=8, min_parent_node_size=2, min_child_node_size=2)

    tr.print_tree()


   
