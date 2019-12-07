# BA 505 Final Project
## Fall 2019
__Dylan Meyer__

## Analysis of Itunes App Data
Cleansing, analysis, and intro modeling of app data from 
<a href="https://www.kaggle.com/tristan581/17k-apple-app-store-strategy-games" target="_blank">Kaggle.com</a>

The Data appears as follows:

| Feature Name                 | Definition                                                                        | Key |
|------------------------------|-----------------------------------------------------------------------------------|-----|
| URL                          | _URL location of the app_                                                         | N/A |
| ID                           | _ID Specific to the App_                                                          | N/A |
| Name                         | _Name of the App_                                                                 | N/A |
| Subtitle                     | _App Subtitle_                                                                    | N/A |
| Icon URL                     | _URL for the icon representing the app_                                           | N/A |
| Average User Rating          | _Average Rating given by users for the app_                                       | N/A |
| User Rating Count            | _Number of ratings submitted for the app_                                         | N/A |
| Price                        | _Cost to purchase the app_                                                        | N/A |
| In-app Purchases             | _Dollar amount of the potential items which can be purchased from within the app_ | N/A |
| Description                  | _Description of the App_                                                          | N/A |
| Developer                    | _Studio who developed the App_                                                    | N/A |
| Age Rating                   | _Intended Audience age for the App_                                               | N/A |
| Languages                    | _Languages supported within the App_                                              | N/A |
| Size                         | _Memory needed on device in order to install the App_                             | N/A |
| Primary Genre                | _Main Genre of the App_                                                           | N/A |
| Genres                       | _All genres the App can be classified as_                                         | N/A |
| Original Release Date        | _Date of initial release of the App_                                              | N/A |
| Current Version Release Date | _Latest update release date of the App_                                           | N/A |


## Overview
An investigation of this data was conducted in order to inform iPhone and iPad users about the 
Apps they use, and those which they may wish to either purchase or install in the future. 

Additionally, there are many scenarios in which an adult has a phone or iPad which they 
share with their children. By leveraging the data provided, recommendations can be made concerning
additional apps to install based on the preferences of those using their device apart from themselves.

#### Questions for Investigation
* Are the ratings of applications related to their price?
* Can new recommended Apps be found based on the icon images of the Apps which they have already installed?
* Do the descriptions of Apps have different terms based on ratings and reviews?
* Can Apps be clustered into groups which can be leveraged to find similar apps?

## Project Structure
This git repo contains several different directories, each with their own purpose
* cleansing
    - data - folder containing data referenced and created
    - notebooks - Initial notebooks and rough exploration work
* scripts
    - Package with helper and display functions for the demo
* predictions
	- Anything used for predictions related to the App data including clustering and cluster visualizations
