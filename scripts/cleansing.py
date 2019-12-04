import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re
import requests
from PIL import Image
from io import BytesIO
from math import ceil
from wordcloud import WordCloud
from difflib import SequenceMatcher
import spacy
from difflib import SequenceMatcher

pd.set_option('max_colwidth', 100)
pd.set_option('display.max_columns', 50)

class cleanse():
    """ 
    The cleanse class contains all of the functionality needed for the demo. This includes descriptive statistics,
    dynamic image recommendations, and charts summarizing App clusters. 
    """
    def __init__(object):
        print('module loaded')
    def visualize_raw_dat_ratings(self, dat):
        """ 
        Visualize all of the ratings through various plots.

        Parameters: 
        dat (df): initial App dataframe

        Returns: 
        void: However, will produce the required graphs inline

        """

        # Ratings Plots
        print('Plots for Ratings Data')

        # Shown as binned ratings
        bins = [0,1,2,3,4,5]
        pd.cut(dat['Average User Rating'], bins).value_counts().plot(kind = 'bar')
        plt.xlabel('Scores')
        plt.ylabel('Count')
        plt.title('Scores Histogram')
        plt.show()

        #Show all ratings
        plt.hist(dat['Average User Rating'].dropna())
        plt.xlabel('Scores')
        plt.ylabel('Count')
        plt.title('Scores Histogram')
        plt.show()
        
    def visualize_raw_dat_rating_cnt(self, dat):
        """ 
        Visualize the number of ratings

        Parameters: 
        dat (df): initial App dataframe

        Returns: 
        void: However, will produce the required graphs inline

        """
        print('Number of Reviews')

        plt.hist(dat['User Rating Count'].dropna())
        plt.show()

        print('There is a long-tail to the data, so a subset will be shown as well')
        print('~80% of the data is pictured below')
        plt.hist(dat[dat['User Rating Count'] < 500].dropna(subset = ['User Rating Count'])['User Rating Count'])
        plt.show()


    def visualize_raw_dat_cost(self, dat):
        """ 
        Visualize how the cost data effects ratings.

        Parameters: 
        dat (df): initial App dataframe

        Returns: 
        void: However, will produce the required graphs inline

        """
        print('Cost of Games')

        plt.hist(dat['Price'].dropna())
        plt.xlabel('Price')
        plt.ylabel('Count')
        plt.title('Price Histogram')
        plt.show()

        print('Previous graph also has a long-tail. Will narrow the scope again')
        print('~99% of results in following graph')

        plt.hist(dat[(dat['Price'] < 15)]['Price'].dropna())
        plt.xlabel('Price')
        plt.ylabel('Count')
        plt.title('Price of Apps')
        plt.show()

        print('''84% of apps are free, skewing the results further. 
            Looking into only non-free apps for a clearer picture on what remains''')

        plt.hist(dat[(dat['Price'] < 15) & (dat['Price'] != 0)]['Price'].dropna())
        plt.xlabel('Price')
        plt.ylabel('Count')
        plt.title('Price of non-free Apps')
        plt.show()

    def autolabel(self, rects, counts):
        """
            Attach a text label above each bar displaying its height
        """

        '''
            Got this from 
            https://stackoverflow.com/questions/30228069/how-to-display-the-value-of-the-bar-on-each-bar-with-pyplot-barh

            Altered it to but whatever text I wanted on the bar instead of the height
        '''
        for x, rect in zip(counts, rects):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., height,
                    'n = ' + str(x),
                    ha='center', va='bottom')

    def prep_visualize_cost(self, dat):
        """ 
        Prep the input dataframe to visualize differences in ratings based on cost of the App.

        Parameters: 
        dat (df): initial App dataframe

        Returns: 
        void: However, will produce the required graphs inline

        """
        # Assume if null then there are no in-app purchases since all others have a value
        dat['In-app Purchases'].fillna('0', inplace = True)

        # An app is only actually free if you cant buy anything once in it
        dat['true_free'] = (dat.Price == 0) & (dat['In-app Purchases'] == '0')

        # Look at only if there is a base cost first and see if rating difference exists

        print('Investigate differences in ratings by free vs. paid App')
        plt.hist(dat[dat.true_free == True]['Average User Rating'].dropna().values, label='Free', color = 'green')
        plt.hist(dat[(dat.true_free == False) & (dat.Price != 0)]['Average User Rating'].dropna().values, label='Paid')

        plt.legend(loc='upper right')
        plt.show()

        print('Many Apps included pain content within them. These should not truly be considered "free"')

        plt.hist(dat[dat.true_free == False]['Average User Rating'].dropna().values, label='Paid')
        plt.hist(dat[dat.true_free == True]['Average User Rating'].dropna().values, label='Free', color = 'green')

        plt.legend(loc='upper right')
        plt.show()

        print('Which are truly best?')

        non_true_free_rating = dat[dat.true_free == False]['Average User Rating'].dropna().mean()
        in_app_purchases = dat[(dat.true_free == False) & (dat.Price == 0)]['Average User Rating'].dropna().mean()
        true_free_rating = dat[dat.true_free == True]['Average User Rating'].dropna().mean()
        zero_cost_rating = dat[dat.Price == 0]['Average User Rating'].dropna().mean()
        paid_rating = dat[dat.Price != 0]['Average User Rating'].dropna().mean()

        non_true_free_rating_cnt = dat[dat.true_free == False]['Average User Rating'].dropna().shape[0]
        in_app_purchases_cnt = dat[(dat.true_free == False) & (dat.Price == 0)]['Average User Rating'].dropna().shape[0]
        true_free_rating_cnt = dat[dat.true_free == True]['Average User Rating'].dropna().shape[0]
        zero_cost_rating_cnt = dat[dat.Price == 0]['Average User Rating'].dropna().shape[0]
        paid_rating_cnt = dat[dat.Price != 0]['Average User Rating'].dropna().shape[0]

        plt.bar(['Zero Cost','Paid App'],
                [zero_cost_rating,paid_rating])
        plt.ylim((3.8,4.18))
        plt.xticks(rotation = 'vertical')
        plt.show()

        plt.bar(['Truly Free','Any Cost'],
                [true_free_rating,non_true_free_rating])
        plt.ylim((3.8,4.18))
        plt.xticks(rotation = 'vertical')
        plt.show()

        plt.bar(['Zero Cost','Truly Free','Paid App','Any Cost','In app-purchases'],
                [zero_cost_rating,true_free_rating,paid_rating,non_true_free_rating, in_app_purchases])
        plt.ylim((3.8,4.18))
        plt.xticks(rotation = 'vertical')
        plt.show()

        N = 5

        ind = np.arange(N)  # the x locations for the groups
        width = 0.35       # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(ind, [zero_cost_rating,true_free_rating,paid_rating,non_true_free_rating, in_app_purchases], 
                        width, color='r')


        # add some text for labels, title and axes ticks
        ax.set_ylabel('Scores')
        ax.set_title('Scores by group and gender')
        ax.set_xticks(ind + width / 2)
        ax.set_xticklabels(['Zero Cost','Truly Free','Paid App','Any Cost','In app-purchases'])

        plt.ylim((3.85,4.18))

        counts = [zero_cost_rating_cnt,true_free_rating_cnt,paid_rating_cnt,non_true_free_rating_cnt, in_app_purchases_cnt]

        for x, rect in zip(counts, rects1):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., height,
                    'n = ' + str(x),
                    ha='center', va='bottom')

        plt.show()

    def time_since_update_stats(self, dat, num_bins = 10):
        '''
            Previous steps needed to make this work:

            dat['curr_release_date'] = pd.to_datetime(dat['Current Version Release Date'], format="%d/%m/%Y")
            dat['time_since_update'] = (np.datetime64('today') - dat['curr_release_date']).astype('timedelta64[D]')

        '''

        print('Loading previously parsed file from ../data/parsed_data.parquet')

        #dat = pd.read_parquet('../cleansing/data/parsed_data.parquet')

        dat.time_since_update.hist()
        plt.show()

        print('Segmenting data with ' + str(num_bins) + ' bins')
        dat['time_since_update_bin'] = pd.cut(dat.time_since_update, bins=num_bins, labels=[i for i in range(num_bins)])

        dat.groupby('time_since_update_bin')['Average User Rating'].mean().plot(kind = 'bar')
        plt.show()
        
    def desc_cleanser(self, txt):
        """ 
        Clean up text to remove unicode and non alpha-numeric

        Parameters: 
        txt (str): text to be cleansed

        Returns: 
        str: Cleansed Text

        """
        # New line issues
        txt = re.sub(r'\\n', r' ', txt)
        # Unicode cleanse
        txt = re.sub(r'\\u[\d]{4}', r'', txt)
        # Remaining unicode cleanse
        txt = re.sub(r'\\{1,2}\S+', r' ', txt)
        # Remove remaining non-alphanumeric and spaces
        txt = ''.join([i for i in txt if i.isalnum() or i.isspace() or i in ['.','?','!']])
        # Remove more than a single space
        txt = re.sub(r'\s+', r' ', txt)

        return txt
    
    def similar(self, a, b):
        return SequenceMatcher(None, a, b).ratio()
    
    def gather_names(self, df,lst,col):
        lst_return = []
        for i in lst:
            lst_return.append(df.iloc[i][col])
        return lst_return
    
    def dif_percentage_lst(self, first,lst):
        '''
        Pixel-wise differences between images
        Narrow the scope for images by resizing to be smaller for faster computations

        Parameters: 
        first (bytes): byte representation of the Image you want to find similar Images to
        lst (list): list of other bytes objects to compare the initial 'first' object to 

        Returns: 
        list: list of similarity scores for images

        '''

        i1 = Image.open(BytesIO(first)).resize((50,50))
        hold = []
        for idx,i2 in enumerate(lst):
            i2 = Image.open(BytesIO(i2)).resize((50,50))
            if i1.mode != i2.mode:
                hold.append(100)
                continue
            assert i1.size == i2.size, "Different sizes."

            pairs = zip(i1.getdata(), i2.getdata())

            #Manhatten distance for images

            if len(i1.getbands()) == 1:
                # for gray-scale jpegs
                dif = sum(abs(p1-p2) for p1,p2 in pairs)
            else:
                dif = sum(abs(c1-c2) for p1,p2 in pairs for c1,c2 in zip(p1,p2))

            ncomponents = i1.size[0] * i1.size[1] * 3

            hold.append((dif / 255.0 * 100) / ncomponents)
        return hold

    
    def visualize_image_sim(self, dat):
        '''
        Guided walkthrough which gathers similar images and presents them to the user.
        
        Parameters: 
        dat (df): parsed dataframe to extract the similar images from

        Returns: 
        void: all operations return images/graphs inline

        '''
        
        app = input('Please Specify an App')
        print('---' * 8)

        ratio = dat.Name.apply(lambda x: self.similar(app,x))
        top_3 = list(ratio.sort_values(ascending=False).index[0:3])
        top_3_names = dat.iloc[top_3].Name.values


        app_refine = input('The most similar items in this set are ' + 
                          ', '.join(top_3_names) +' please specify which you are interested in')

        print('---' * 8)
        print('Looking for ' + app_refine)
        print('---' * 8)

        while (app_refine not in top_3_names) and (app_refine != 'end'):
            app_refine = input('Exact match not found, please specify the exact app' + 
                          ', '.join(top_3_names))

        if app_refine == 'end':
            print('ending process')
            return


        print('Icon for chosen App will be Displayed in pop-up')

        curr_image = Image.open(BytesIO(dat[dat.Name == app_refine].image_bytes.values[0]))

        curr_image.show()

        print('---' * 8)
        print('Gathering Similar Images')
        print('---' * 8)

        res = self.dif_percentage_lst(dat[dat.Name == app_refine].image_bytes.values[0], dat.dropna().image_bytes.values)


        best_images = self.gather_names(dat.dropna(),list(pd.Series(res).sort_values().index),'Name')[1:4]

        print('Top image similarities are ' + ', '.join(best_images))
        print('---' * 8)

        view = input('Would you like to view the images? Y/N')
        print('---' * 8)

        if view == 'Y':
            for i in best_images:
                similar_image = Image.open(BytesIO(dat[dat.Name == i].image_bytes.values[0]))
                similar_image.show()

            print('Images displayed')
    
        return
    
    def show_wordcloud(self, data, title = None):
        '''
        Display wordcloud for the given text data
        
        Parameters: 
        data (str): string data for the wordcloud 

        Returns: 
        void: Wordcloud displayed inline

        '''
        wordcloud = WordCloud(
            background_color='white',
            max_words=200,
            max_font_size=40, 
            scale=3,
            random_state=6
        ).generate(str(data))

        fig = plt.figure(1, figsize=(12, 12))
        plt.axis('off')
        if title: 
            fig.suptitle(title, fontsize=20)
            fig.subplots_adjust(top=2.3)

        plt.imshow(wordcloud)
        plt.show()
