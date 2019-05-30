# RailsReactML
Implementing the classification problem: dublicate, modification, similar classification.

Console application. Importing the sys library to work with console parameters.
    
Launch the program with next syntax:
    
python3 solution.py [1] [2] [3] [4] [5] - to define coefficients
--- OR ---
python3 solution.py [1] [2] - to compile without coefficients (the program will use the default ones, the most optimal)
    
--- ANOTHER OPTION ---
    
./solution.py [1] [2] [3] [4] [5] - to define coefficients
--- OR ---
./solution.py [1] [2] - to compile without coefficients (the program will use the default ones, the most optimal)
    
Parameters:
    [1] Algorithm we use, options: {ahash, dhash, phash}
    [2] The working directory, where we store the images.
    {[3], [4], [5]} The scale parameters, for dublicate/modification/similarity finding, that we will use in our program. 
    [3] - dublicate (the most compatible input values are {0, 1})
    [4] - modifications: (the most compatible input values are {6, 7, 8})
    [5] - similar: (the most compatible input values are {13, 14, 15, 16})
