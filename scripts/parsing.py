import pandas as pd
import requests
import spacy
import numpy as np
import re

class parse():
    """ 
    The parse class contains all of the functionality that was used to prep the data showing in the demo notebook.
    """
    def __init__(object):
        print('module loaded')
        
    def load_raw(self):
        '''
        Load in raw data from where it currently is in repo
        
        Returns:
        df: raw dataframe
        '''
        raw_data = pd.read_csv('./cleansing/data/appstore_games.csv')
        raw_data.drop_duplicates(subset='ID', inplace = True)
        return raw_data
    
    def inferred_cols(self, dat):
        '''
        Parameters: 
        dat (df): dataframe with a few columns which need to be updated

        Returns: 
        df: dataframe with updated values
        '''
        # Null for purchases implies 0
        dat['In-app Purchases'].fillna('0', inplace = True)
        # If both forms of purchasing are 0 then it is truly 0
        dat['true_free'] = (dat.Price == 0) & (dat['In-app Purchases'] == '0')
        dat['curr_release_date'] = pd.to_datetime(dat['Current Version Release Date'], format="%d/%m/%Y")
        # Some of the dates are strange, so use today's date to check
        dat['time_since_update'] = (np.datetime64('today') - dat['curr_release_date']).astype('timedelta64[D]')
        dat['time_since_update_bin'] = pd.cut(dat.time_since_update, bins=10, labels=[i for i in range(10)])
        dat['parsed_desc'] = dat.Description.apply(self.desc_cleanser)
        
        return dat
    
    def desc_cleanser(self, txt):
        '''
        Utility function to clean the description field of dataframe
        
        Parameters: 
        txt (str): string which needs to be cleansed

        Returns: 
        str: string which has had noise removed
        '''
        
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
    
    def get_image_bytes(self, df, col):
        '''
        Use requests library to get the bytes from a url pointing at an image
        
        Parameters: 
        df (df): dataframe with urls in it
        col (str): column which contains the urls

        Returns: 
        list: list of the contents returned from the url

        This take a VERY long time to run. Try to minimize number of times this needs to run
        '''
        images = []
        for i in df[col].values:
            try:
                response = requests.get(i)
                images.append(response.content)
            except Exception as e:
                images.append(None)
        return images
    
    def prep_string_lists(self, df, cols = ['Languages','Genres']):
        '''
        Some columns are strings which are actually lists and need to be cleansed.
        
        Parameters: 
        df (df): initial dataframe with columns that have string lists
        cols (list): columns which need to be transformed

        Returns: 
        df: dataframe with columns updated to reflect proper data type
        
        '''
        
        for i in cols:
            df[i] = df[i].apply(lambda x: x.split(', ') if x is not None else None)
        return df
    
    def desc_parsing(self, df, col):
        '''
        Utilize Spacy for lemmas, part of speech, and entities
        
        Parameters: 
        df (df): dataframe which contains textual information
        col (str): the column which contains the text we want to extract information about

        Returns: 
        list: list of the contents returned from the url
        '''
        nlp = spacy.load("en_core_web_sm")
        df['spacy_text'] = df[col].apply(lambda x: [i for i in nlp(x) if (i.is_stop == False 
                                                                                                and i.is_alpha == True)])
        df['desc_parsed_text'] = df['spacy_text'].apply(lambda x: [i.lemma_.lower() for i in x])
        df['desc_parsed_pos'] = df['spacy_text'].apply(lambda x: [i.pos_ for i in x])
        df['desc_parsed_ents'] = df['spacy_text'].apply(lambda x: [i.ent_type_ for i in x if i.ent_type_ != ''])
        return df

    
    def produce_demo_dfs(self, test):
        '''
        Parameters: 
        test (bool): Whether or not this is testing functionality or running all of it

        Returns: 
        void: Calls other methods, producing graphs and updating data
        '''
        raw_data = self.load_raw()
        dat = raw_data.copy()
        
        if test:
            dat = dat.iloc[0:5]

        dat = self.inferred_cols(dat)
        
        print('Loading images, this will take awhile')
        dat['image_bytes'] = self.get_image_bytes(dat, 'Icon URL')
        print('images loaded')
        save_str = ''
        if test:
            save_str = './cleansing/data/TEST_parsed_data.parquet'
            dat.to_parquet(save_str)
        else:
            save_str = './cleansing/data/parsed_data.parquet'
            dat.to_parquet(save_str)
        print('Interim DF saved at ' + save_str)
              
        text_sim_set = dat.copy()
        text_sim_set = self.prep_string_lists(df = text_sim_set)
        text_sim_set = self.desc_parsing(text_sim_set, 'parsed_desc')
        
        if test:
            save_str = './cleansing/data/TEST_text_sim_set.csv'
            text_sim_set.to_csv(save_str)
        else:
            save_str = './cleansing/data/text_sim_set.csv'
            text_sim_set.to_csv(save_str)
        print('Final DF saved at ' + save_str)
        
    def run_all(self, test = False):
        self.produce_demo_dfs(test)