#Evi Nikolaidou - S15129366
#'Do the number of vehicles and casualties in a traffic accident affect accident severity rating?' 

import numpy as np
import pandas as pd
import time

from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

start = time.time()

df = pd.read_csv('Accidents2016.csv')
df.drop('Accident_Index', 1, inplace=True)
df.drop('Location_Easting_OSGR', 1, inplace=True)
df.drop('Location_Northing_OSGR', 1, inplace=True)
df.drop('Longitude', 1, inplace=True)
df.drop('Latitude', 1, inplace=True)
df.drop('Police_Force', 1, inplace=True)
df.drop('Local_Authority_(District)', 1, inplace=True)
df.drop('Local_Authority_(Highway)', 1, inplace=True)
df.drop('1st_Road_Class', 1, inplace=True)
df.drop('1st_Road_Number', 1, inplace=True)
df.drop('Junction_Detail', 1, inplace=True)
df.drop('Junction_Control', 1, inplace=True)
df.drop('2nd_Road_Class', 1, inplace=True)
df.drop('2nd_Road_Number', 1, inplace=True)
df.drop('Pedestrian_Crossing-Human_Control', 1, inplace=True)
df.drop('Pedestrian_Crossing-Physical_Facilities', 1, inplace=True)
df.drop('Special_Conditions_at_Site', 1, inplace=True)
df.drop('Carriageway_Hazards', 1, inplace=True)
df.drop('Urban_or_Rural_Area', 1, inplace=True)
df.drop('Did_Police_Officer_Attend_Scene_of_Accident', 1, inplace=True)
df.drop('LSOA_of_Accident_Location', 1, inplace=True)
df.drop('Time', 1, inplace=True)
df.drop('Date', 1, inplace=True)

df.drop('Road_Type', 1, inplace=True)
df.drop('Speed_limit', 1, inplace=True)
df.drop('Light_Conditions', 1, inplace=True)
df.drop('Weather_Conditions', 1, inplace=True)
df.drop('Road_Surface_Conditions', 1, inplace=True)


df.dropna(inplace=True)
X = np.array(df.drop('Accident_Severity',1)) 
y = np.array(df['Accident_Severity'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)


# Initialize our classifier
gnb = GaussianNB()

# Train our classifier
model = gnb.fit(X_train, y_train)

# Make predictions
preds = gnb.predict(X_test)
print(preds)


#To visualize the data to a grafic figure
plt.title('Accident Severity vs Number of vehicles and casualties')
plt.ylabel('Accident Severity')
plt.xlabel('Number of vechicles and casualities')
plt.plot(y_test, preds,color = 'green')
plt.show()
#to show runtime and accuracy score
end = time.time()
print('Time taken was', end - start, 'seconds.')
print('The accuracy score is:', accuracy_score(y_test, preds))

