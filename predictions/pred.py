import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

class pred():
    """ 
    The pred class contains all of the functionality that was used to add predictions as shown in the demo notebook.
    """
    def __init__(object):
        print('module loaded')
        
    def gen_feats(self, df):
        '''
        Parameters: 
        df (df): dataframe which needs features to be generated

        Returns: 
        df: dataframe with features
        '''
        feats = pd.concat([df[['Average User Rating',
       'User Rating Count', 'Price','true_free','time_since_update']],
                  pd.get_dummies(df['Languages'].apply(pd.Series).stack()).sum(level=0),
                 pd.get_dummies(df['Genres'].apply(pd.Series).stack()).sum(level=0)],
                  axis = 1)
        # Previously checked the null values and feel ok with filling them all with 0 
        # as null for these columns implies 0
        feats.fillna(value=0, inplace=True)
        return feats
    
    def distortion_plot(self, df):
        '''
        Parameters: 
        df (df): dataframe wto find optimal number of clusters for

        Returns: 
        void: plots distortion values to pick optimal ampunt of clusters
        '''
        distortions = []
        for i in range(1, 10):
            km = KMeans(
                n_clusters=i, init='random',
                n_init=25, max_iter=300,
                tol=1e-04, random_state=0
            )
            km.fit(df)
            distortions.append(km.inertia_)
        plt.plot(range(1, 10), distortions, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Distortion')
        plt.show()
        
    def fit_clusters(self, df):
        '''
        Parameters: 
        df (df): dataframe to cluster

        Returns: 
        list: list containing the clusters based on df rows
        '''
        km = KMeans(
            n_clusters=5, init='random',
            n_init=25, max_iter=300, 
            tol=1e-04, random_state=6
        )
        y_km = km.fit_predict(df)
        
        return y_km
        
    def visualize_dat(self, df):
        '''
        Make a relplot for the purposes of visualizing the differences based on clusters
        
        Parameters: 
        df (df): dataframe to use for visualization

        Returns: 
        void: prints chart inline
        '''
        sns.relplot(x="User Rating Count", y="Average User Rating", hue="cluster", size="Price",
            sizes=(40, 400), alpha=.5,
            palette = sns.color_palette("Set2", n_colors = df[df['User Rating Count'] < 100000]['cluster'].nunique()),
            height=6, data=df[df['User Rating Count'] < 100000])
        plt.show()
        
    def add_pred(self, dat_path = './cleansing/data/text_sim_set-11-30-19.csv'):
        '''
        Parameters: 
        dat_path (str): where the data needs to be loaded from

        Returns: 
        void: but will add cluster data to df and save it
        '''
        print('Reading Data')
        text_sim_set = pd.read_csv(dat_path)
        print('Generating Features')
        feats = self.gen_feats(text_sim_set)
        print('Plotting Distortion')
        self.distortion_plot(feats)
        feats['cluster'] = self.fit_clusters(feats)
        self.visualize_dat(feats)
        
        # self.distortion_plot(feats[list(feats.columns)[:1] + list(feats.columns)[2:]])
        feats['cluster'] = self.fit_clusters(feats[list(feats.columns)[:1] + list(feats.columns)[2:]])
        self.visualize_dat(feats)
        
          