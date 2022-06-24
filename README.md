# Neural Network NBA Player Project
The goal of this project was to predict the points per game and games played based on NBA player age, height, and weight. The data consists of the NBA player data from 1996-2019.

Player points, age, height, weight, and games played were compressed into clustered ranges to improve the accuracy of the model (Ex. 1 = 18.0 through 23.0 in years/ age of player)

The variables season, draft year, draft round, and team abbreviation were label encoded to change them to categorical numbers (Ex. 1 = 1996-97 season).

The features that were used in this neural network to predict games played and point range were age, player height, team abbreviation, draft year, and season. 

The model produced a 76.5% accuracy. 
