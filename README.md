# BYORS (Build Your Own Recommednation System)

This repository will allow you to create your own movie recommendation system! Utilize SQL and Python to make recommendations based on genre,year,movie tags, and more! 
There is no right answer, so the possibilites are endless. Check out my project to see how I made my Hybrid Recommendation System.

## Table of Contents
- [Data](#Data)
- [Technology](#Technology)
- [Features](#Features)
- [Goal](#Goal)

## Data
The database file consist of 3 tables

MOVIES
-MovieID (INT)
-Title (TEXT)
-Genres (TEXT)

RATINGS
-UserID (INT)
-MovieID (INT)
-Tag (TEXT)
-Timestamp (NUMERIC)

TAGS
-UserID (INT)
-MovieID (INT)
-Tag (TEXT)
-Timestamp (NUMERIC)

You can read the database file into DBeaver and work from there.

Then once you have done all the querying in SQL to make the desirebale tables, then you can use the "sqllite3" package in Python to export those to Pycharm. 

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

## Goal
This project was a great way for me to get creative with a very ambigious and general task, build a movie recommendation system. There is no right answer, so this project is a way to really get creative and see if your system stacks up against Netflix. 

One part of this project I hope to provide value to others is the construction of the Hybrid System. When I worked on this project, I spent a lot of time looking for ways to combine a collaborative system with a content-based, but never found anything online concretly answering my questions. 

I hope this provides a great way for people to get exposed to SQL and Python centered projects, and would love to see how your final results stack up against mine! 



