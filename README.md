# BYORS (Build Your Own Recommednation System)

This repository will allow you to create your own movie recommendation system! Utilize SQL and Python to make recommendations based on genre,year,movie tags, and more! 
There is no right answer, so the possibilites are endless. Check out my project to see how I made my Hybrid Recommendation System.

## Table of Contents
- [Data](#Data)
- [Technology](#Technology)
- [Features](#Features)
- [Process](#Process)
- [Goal](#Goal)

## Data
The database file consist of 3 tables

MOVIES
- MovieID (INT)
- Title (TEXT)
- Genres (TEXT)

RATINGS
- UserID (INT)
- MovieID (INT)
- Tag (TEXT)
- Timestamp (NUMERIC)

TAGS
- UserID (INT)
- MovieID (INT)
- Tag (TEXT)
- Timestamp (NUMERIC)

You can read the database file into DBeaver and work from there.

Then once you have done all the querying in SQL to make the desirebale tables, then you can use the "sqlite3" package in Python to export those to Pycharm. 

## Technology

Software
- DBeaver
- PyCharm

Python Packages
- scipy.spatial import distance
- numpy
- pandas 
- sqlite3
- sklearn.metrics.pairwise import cosine_similarity
- collections import Counter
- ast
  
## Skills Demonstrated 
- SQL
- Python
- Content-Based Filtering 
- Collaborative Filtering
- Hybrid System Construction
- Tableau

## Process
This a simple step by step of what I did and where.

### Step 1: Data Exploration/Clean the Data in DBeaver
- Download the Database in Dbeaver
- Use SQL to remove rows with null values
- Create custom tables needed for Content and Collab filtering
- Create Visualization of the movies based on thier rating and popularity
  ![Quadrant](https://github.com/MaxGhasemi/MoviesDataBase/assets/120604692/c16fdc23-9c3c-434b-974b-6a91e2d937a1)


### Step 2: Collaborative Filtering 
- Create a similarity matrix and compute cosine similarity
- Convert matrix to dataframe, find 5 most similar users
- Create a for loop to get every movie those similar users have rated over a 4 or higher and that at least 3 of those similar user's have seen.
- Extract the first 5 movies to the final recommendations list

### Step 3: Content Based Filtering 
- Create a similarity matrix and compute cosine similarity
- Convert matrix to dataframe
- Create a new dataframe that has the 10 most similar movies for each movie title

### Step 4: Creating the Hybrid
- Load the content and collaborative dataframes
- Start a for loop
  - Get a list of every movie watched by that user
  - Get their highest rated movie
  - Get the first 5 names in the collaborative list ( Already Filtered for unseen movies)
  - Add movie titles to the final recommendations list
  - If no movies are recommended in prior step, use the user's top movie for content based recommendations
  - If movies are recommended, find the one with the highest rating and popularity score and use that title for content based recommendations
  - Add, at most, the 5 most similar movies to the given title to the final recommendations list
  - If the list is not at 10 movies, fill in the list with unseen High Rated/High Popularity Scored movies
- See Hybrid Visualization folder to get visual on how this works
![Hybrid](https://github.com/MaxGhasemi/MoviesDataBase/assets/120604692/69a8c2b7-1024-4688-8ccf-efc351165460)
![Hybrid2](https://github.com/MaxGhasemi/MoviesDataBase/assets/120604692/eec70af7-f159-43c8-acf3-4063f1f1c92c)
![Hybrid3](https://github.com/MaxGhasemi/MoviesDataBase/assets/120604692/44bdcea0-34e7-45c5-96fa-a78958907d9e)

## Goal
This project was a great way for me to get creative with a very ambigious and general task, build a movie recommendation system. There is no right answer, so this project is a way to really get creative and see if your system stacks up against Netflix. 

One part of this project I hope to provide value to others is the construction of the Hybrid System. When I worked on this project, I spent a lot of time looking for ways to combine a collaborative system with a content-based, but never found anything online concretly answering my questions. 

I hope this provides a great way for people to get exposed to SQL and Python centered projects, and would love to see how your final results stack up against mine! 



