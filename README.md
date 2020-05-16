# text-genre-classifier
Classify any text into various genres based on the plot summary of the text

## The model is trained and tested on the following dataset
http://www.cs.cmu.edu/~ark/personas/

## The model was again ran on IMDB movie dataset to test accuracy
https://www.kaggle.com/PromptCloudHQ/imdb-data


Sample Predictions on IMDB Dataset
```
John Wick------------->Action,Crime,Thriller
An ex-hitman comes out of retirement to track down the gangsters that took everything from him.
Predicted--> [('Crime Fiction',)]

Pirates of the Caribbean: At World's End------------->Action,Adventure,Fantasy
Captain Barbossa, Will Turner and Elizabeth Swann must sail off the edge of the map, navigate treachery and betrayal, find Jack Sparrow, and make their final alliances for one last decisive battle.
Predicted--> [('Action/Adventure',)]

Dead Awake------------->Horror,Thriller
A young woman must save herself and her friends from an ancient evil that stalks its victims through the real-life phenomenon of sleep paralysis.
Predicted--> [('Horror',)]

Lowriders------------->Drama
A young street artist in East Los Angeles is caught between his father's obsession with lowrider car culture, his ex-felon brother and his need for self-expression.
Predicted--> [('Drama', 'Indie')]

** NOT PREDICTED ON LOW ACCURACY**
Zootopia------------->Animation,Adventure,Comedy
In a city of anthropomorphic animals, a rookie bunny cop and a cynical con artist fox must work together to uncover a conspiracy.
Predicted--> [()]

** FALSELY PREDICTED IN SOME PLOTS**
The Autopsy of Jane Doe------------->Horror,Mystery,Thriller
A father and son, both coroners, are pulled into a complex mystery while attempting to identify the body of a young woman, who was apparently harboring dark secrets.
Predicted--> [('Drama',)]



```
Sample 
# Desclaimer 
The threshold value is set to 0.5
Predictions with less than 0.5 accuracy won't be detected
