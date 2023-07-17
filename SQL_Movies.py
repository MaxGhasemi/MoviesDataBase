# Load all the necessary packages

from scipy.spatial import distance
import numpy as np
import pandas as pd
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import ast


# connect to the Movies database. You need to use the path on your computer
# Will need to change below code to access SQL
database = r"C:\Users\maxgh\OneDrive\Documents\Masters\BSAN 780\SQL Review\Final Project Movies SQL Folder\Movies.db"
conn = sqlite3.connect(database)
cur = conn.cursor()

# you can use SQL to query SQLite tables from inside Python
cur.execute("SELECT * FROM final_min_max")

# Convert the One Hot Encoded Genres Table to a Pandas Dataframe
genres_dataframe = pd.DataFrame(cur.fetchall())

# View Dataframe
print(genres_dataframe)

# Add Column Names
genres_dataframe.columns= ['userID','Scaled_Action','Scaled_Adventure','Scaled_Animation','Scaled_Children','Scaled_Comedy','Scaled_Crime','Scaled_Documentary','Scaled_Drama','Scaled_Fantasy','Scaled_Film_Noir','Scaled_Horror','Scaled_IMAX','Scaled_Musical','Scaled_Mystery','Scaled_Romance','Scaled_Sci_Fi','Scaled_Thriller','Scaled_War','Scaled_Western']

# View Dataframe
print(genres_dataframe)

# you can export a dataframe to a csv file
#genres_dataframe.to_csv('genres.csv')

# select only the genre preference columns
genre_prefs = genres_dataframe.iloc[:, 1:].values

# compute the cosine similarity matrix
genres_similarity_matrix = cosine_similarity(genre_prefs)

# Print Similarity Matrix for Genres
print(genres_similarity_matrix)

# Print Shape of Similarity Matrix for Genres
print(genres_similarity_matrix.shape)

# Above Output is right since there are 610 users in the Rating Table (Which we ultimately are using)

# Fix Problem of Comparing people against themselves
np.fill_diagonal(genres_similarity_matrix, 0)

# Print Similarity Matrix for Genres to make sure diagonals are 0
print(genres_similarity_matrix)


# Convert Similarity Matrix to Dataframe
genres_sim_df = pd.DataFrame(genres_similarity_matrix, columns=range(1, len(genres_dataframe)+1))

# Have the Rows start from 1 instead of 0 so the UserId's relate to eachother
genres_sim_df.index=genres_sim_df.index+1

# Print Similarity Matrix for Genres to make sure diagonals are 0
print(genres_sim_df)

# Get the top 10 most similar users based on Genre Preference
genres_top10_match = genres_sim_df.apply(lambda x: x.nlargest(10).index.tolist(), axis=1)

#Print Output
print(genres_top10_match)

# you can export a dataframe to a csv file
#genres_top10_match.to_csv('genres_results.csv')


# Will need to change below code to access SQL
database = r"C:\Users\maxgh\OneDrive\Documents\Masters\BSAN 780\SQL Review\Final Project Movies SQL Folder\Movies.db"
conn = sqlite3.connect(database)
cur = conn.cursor()

# you can use SQL to query SQLite tables from inside Python
cur.execute("SELECT * FROM user1_movies")

# Convert the One Hot Encoded Genres Table to a Pandas Dataframe
user1_dataframe = pd.DataFrame(cur.fetchall())

user1_dataframe.to_csv('user1.csv')



















########################### Decided Not to go forward with Genre Profiling because of the lack of diversity in recommendations

# Now that we have the 10 most common users for each user based on genre preference, we can look to rating similarity
# Note this look at the ratings patterns of individuals

# you can use SQL to query SQLite tables from inside Python
cur.execute("SELECT * FROM movie_and_ratings")

# Convert the Ratings table to a Pandas Dataframe
ratings_dataframe = pd.DataFrame(cur.fetchall())

# View Dataframe
print(ratings_dataframe)

# Add Column Names
ratings_dataframe.columns=['title','userID','Rating']

# View Dataframe
print(ratings_dataframe)

# Create a pivot table where rows are users, columns are movies, and values are ratings
ratings_matrix = ratings_dataframe.pivot_table(index='userID', columns='title', values='Rating')

# View Matrix
print(ratings_matrix)

# Replace NaN with 0
ratings_matrix = ratings_matrix.fillna(0)

# View Matrix
print(ratings_matrix)

# Check the shape, 610 different users, 4925 Movies
ratings_matrix.shape

# Compute the cosine similarity matrix
ratings_similarity_matrix = cosine_similarity(ratings_matrix)

# Print Similarity Matrix for ratings
print(ratings_similarity_matrix)

# Print Shape of Similarity Matrix for ratings
print(ratings_similarity_matrix.shape)

# Above Output is right since there are 610 users in the Rating Table (Which we ultimately are using)

# Fix Problem of Comparing people against themselves
np.fill_diagonal(ratings_similarity_matrix, 0)

# Print Similarity Matrix for ratings to make sure diagonals are 0
print(ratings_similarity_matrix)

# Convert Similarity Matrix to Dataframe
ratings_sim_df = pd.DataFrame(ratings_similarity_matrix, columns=range(1, len(ratings_matrix)+1))

# Print Similarity Matrix for ratings to make sure diagonals are 0
print(ratings_sim_df)

# you can export a dataframe to a csv file
ratings_sim_df.to_csv('ratings_sim_df.csv')

## This chunk below is what she used in class, we will find the 5 most similar users
# this will find the index of the most similar customer
#most_similar_user_list = ratings_sim_df.idxmax(axis=1).tolist()
#print(most_similar_user_list)


# Get the top 5 most similar users based on Ratings
ratings_top5_match = ratings_sim_df.apply(lambda x: x.nlargest(5).index.tolist(), axis=1)

# Print Output
print(ratings_top5_match)

# you can export a dataframe to a csv file
#ratings_top5_match.to_csv('ratings_results.csv')

##################### MOVIE RECOMMENDATION
recommended_movies_rating = [] # Create an empty list for the recommendations
for userID in range(len(ratings_top5_match)): # Create a For Loop to run through each userID
    cur.execute("SELECT title from movie_and_ratings where userID = ? and rating >= 1;", (userID+1,)) # Get every movie the user has seen
    customer_A_movies_watched = [x[0] for x in cur.fetchall()] # Get a list of every User

    # Find the movies that the top 5 most similar users have rated 4 or higher, but that the current user hasn't seen
    similar_users = ratings_top5_match.iloc[userID] # Get the top 5 most similar users for the current user
    similar_users_fav_movies = [] # Create an empty list to store movies rated 4 or higher for similar users
    for similar_user in similar_users: # Loop through each similar user for each userID (5 per user)
        cur.execute("SELECT title from movie_and_ratings where userID = ? and rating >= 4;", (similar_user,)) # Get the movieID of movies with a rating of 4 or higher
        similar_user_fav_movies = [x[0] for x in cur.fetchall()] # Format the values to look clean in a list
        similar_users_fav_movies.extend(similar_user_fav_movies) # Add that list of movies to the list of similar users fav movies
    similar_users_fav_movies_counter = Counter(similar_users_fav_movies) # Add a counter to the list we have made above
    customer_sim_fav_movies = []
    count = 5 # Set the count to 5
    while len(customer_sim_fav_movies) < 5 and count >= 3: # While the list of Movies has not reached 5 reccomendations, reccomend movies with an intersect of at least 3 movies
        movies_with_count = [x for x, c in similar_users_fav_movies_counter.items() if c == count] # Gets list of all movies similar users rated 4 or higher, starts with movies all similar users watched and goes down to 3 until there are 5 reccomendations
        recos = np.setdiff1d(movies_with_count, customer_A_movies_watched) # Find movies Targeted user has not watched
        customer_sim_fav_movies.extend(recos) # List of Movie Recommendation
        count -= 1 # This allows us to keep going down to movies with 4/5 users liking down to a minimum 3/5 users

    recommended_movies_rating.append(customer_sim_fav_movies[:5]) # Adds first 5 movies to recommendation list

print(recommended_movies_rating) # Full Recommendation list


# Convert the list into a dataframe
collab_rec_movies = pd.DataFrame({'user_id': range(1, len(recommended_movies_rating)+1), 'recommended_movies': recommended_movies_rating})

# Print the dataframe
print(collab_rec_movies)

# Print the Rows for A column
collab_rec_movies.loc[2,:]

# you can export a dataframe to a csv file
collab_rec_movies.to_csv('collab_dataframe.csv')


####################

################################### Content Based Filtering

# you can use SQL to query SQLite tables from inside Python
cur.execute("SELECT * FROM final_content_based_movies")

# Convert the Ratings table to a Pandas Dataframe
content_based_dataframe = pd.DataFrame(cur.fetchall())

# View the Dataframe
print(content_based_dataframe)


# Add Column Names
content_based_dataframe.columns=['title', 'action', 'adventure', 'animation', 'children','comedy','crime', 'imax','documentary','drama','fantasy', 'film_noir', 'horror','musical','mystery', 'romance','sci_fi','thriller', 'war','Western','Average_Rating','Scaled_Watched']

# View Dataframe
print(content_based_dataframe)

# Exclude Movie Title
content_based_matrix = content_based_dataframe.iloc[:, 1:].values

# View Matrix
print(content_based_matrix)

# Check the shape, 610 different users, 4925 Movies
content_based_matrix.shape

# Compute the cosine similarity matrix
content_based_similarity_matrix = cosine_similarity(content_based_matrix)

# Print Similarity Matrix for content_based
print(content_based_similarity_matrix)

# Print Shape of Similarity Matrix for content_based
print(content_based_similarity_matrix.shape)

# Above Output is right since there are 610 users in the Rating Table (Which we ultimately are using)

# Fix Problem of Comparing people against themselves
np.fill_diagonal(content_based_similarity_matrix, 0)

# Print Similarity Matrix for content_based to make sure diagonals are 0
print(content_based_similarity_matrix)

# Convert Similarity Matrix to Dataframe
content_based_sim_df = pd.DataFrame(content_based_similarity_matrix, columns=range(1, len(content_based_matrix)+1))

# Print Similarity Matrix for content_based to make sure diagonals are 0
print(content_based_sim_df)

# Connect back Movie Titles
content_based_sim_df = pd.concat([content_based_dataframe['title'], pd.DataFrame(content_based_similarity_matrix)], axis=1)

# Set Row labels as Corresponding Movie Titles
content_based_sim_df = content_based_sim_df.set_index('title')

# Set Column Names as Corresponding Movie Titles
content_based_sim_df.columns = content_based_dataframe['title'].values

# Print Similarity Matrix for content_based to make sure diagonals are 0 and titles are included
print(content_based_sim_df)

# Get the top 10 most similar movies based on content-based filtering
content_based_top10_match = content_based_sim_df.apply(lambda x: x.nlargest(10).index.tolist(), axis=1)
content_based_top10_match = pd.DataFrame(content_based_top10_match.tolist(), index=content_based_sim_df.index)

# Print Output
print(content_based_top10_match)

# you can export a dataframe to a csv file
#content_based_top10_match.to_csv('content_based_results.csv')

#### Hybrid System

# First, Let's Load the Collorative and Content Based Data frames

# Content Based
content = pd.read_csv(r"C:\Users\maxgh\PycharmProjects\Movies_Reccommend\Hybrid Final\content_based_results.csv")
print(content)

# Collab Based
collab = pd.read_csv(r"C:\Users\maxgh\PycharmProjects\Movies_Reccommend\Hybrid Final\collab_dataframe.csv")
print(collab)
# Next we need to get the csv for the Movie Ratings

# you can use SQL to query SQLite tables from inside Python
cur.execute("SELECT * FROM binned_movie_ratings")

# Convert the Ratings table to a Pandas Dataframe
binned_movie_ratings = pd.DataFrame(cur.fetchall())

# View the Dataframe
print(binned_movie_ratings)

binned_movie_ratings = binned_movie_ratings.rename(columns={0: 'title', 1: 'Average Rating', 2: 'Popularity Score'})

# View the Dataframe
print(binned_movie_ratings)
#binned_movie_ratings.to_csv('movie_ratings.csv')

#ratings_df = pd.read_csv(r"C:\Users\maxgh\PycharmProjects\Movies_Reccommend\Hybrid\movie_ratings.csv")

# Print the Data frames

print(content)
print(collab)
print(binned_movie_ratings)

## Add New Table for User Movies

# you can use SQL to query SQLite tables from inside Python
cur.execute("SELECT userID, title FROM movie_and_ratings ORDER BY userID")
# Convert the Ratings table to a Pandas Dataframe
user_movies = pd.DataFrame(cur.fetchall())

# View the Dataframe
print(user_movies)
user_movies = user_movies.rename(columns={0: 'userID', 1: 'movie_watched'})


# Create a New Data Frame with a list for every


# you can use SQL to query SQLite tables from inside Python
cur.execute("SELECT * FROM top_rated_movie")

# Convert the Ratings table to a Pandas Dataframe
top_rated_movie = pd.DataFrame(cur.fetchall())

# View the Dataframe
print(top_rated_movie)
top_rated_movie = top_rated_movie.rename(columns={0: 'title', 1: 'userID', 2: 'Highest_Rating'})

# Rename userID
collab = collab.rename(columns={'user_id': 'userID'})

# Delete Unnamed Column in Collab
collab = collab.drop(collab.columns[0], axis=1)



#############################################33


# Now we have all the dataframes we need to create the hybrid system


# Dataframes needed for Hybrid
print(content)
print(collab)
print(binned_movie_ratings)
print(user_movies)
print(top_rated_movie)

# Test run for user 1 to plan out the structure for the system

# Start off with doing all of this for user1 to test
final_recommendations_1 = []
# Step 1 Create a List of all the movies each User has watched
print(user_movies)

# Create a list for each user
user_movies_grouped = user_movies.groupby('userID')['movie_watched'].agg(list).reset_index()
print(user_movies_grouped)
# Final List of every movie user 1 has watched
user_1_movies_watched = user_movies_grouped.loc[user_movies_grouped['userID'] == 1, 'movie_watched'].values[0]


# Step 2 get the users's highest rated movie
print(top_rated_movie)
user1_top_movie = top_rated_movie[top_rated_movie['userID'] == 1].iloc[0]['title']

# Step 3 Get the first 5 names in the collab list (Already filtered to make sure the user has not seen before)
print(collab)
collab_recommended_movies_1 = ast.literal_eval(collab.loc[collab['userID'] == 1, 'recommended_movies'].values[0]) # Go to the recommended movies column for the user we are looking at
collab_recommended_movies_1_recs = collab_recommended_movies_1[:5] # Return the first 5 values in the list
print(collab_recommended_movies_1_recs)

# Add the Collab Values to the List
final_recommendations_1.extend(collab_recommended_movies_1_recs)

# View the final recommendations list
print(final_recommendations_1)

# Step 4 Check to see if there are any movies in the above list, if not, use the user's top_rated_movie to suggest
if len(final_recommendations_1) >= 1:
    # Find the movie out of the suggested movies that has the highest rating and popularity score
    # Select the Rows in the Binned Ratings Table for each movie selected in the collab
    filtered_df_1 = binned_movie_ratings.loc[binned_movie_ratings['title'].isin(final_recommendations_1)]
    # Find the movie out of the suggested with the highest Popularity Score and Average Rating
    top_movie_out_of_collab = filtered_df_1.sort_values(by=['Popularity Score', 'Average Rating'], ascending=False).iloc[0]['title']
    # Return the Top 5 most similar movies to the 'top_movie'
    full_recommended_content_1 = content.loc[content['title'] == top_movie_out_of_collab, :].values.tolist()[0][1:11]
    # Remove movies that user has already seen.
    unseen_user1_recommended_content_1_recs = [x for x in full_recommended_content_1 if x not in user_1_movies_watched]
    # Return First 5 movies in the unseen movie list
    if len(unseen_user1_recommended_content_1_recs) >= 5:
        user1_final_content_recs = [] # Create an empty list
        for movie in unseen_user1_recommended_content_1_recs:  # Run through each of the movies
            if movie not in final_recommendations_1: # If the movie is not already recommended
                user1_final_content_recs.append(movie) # Add it to the above list
            if len(user1_final_content_recs) == 5: # Add at most 5 movies
                break
        final_recommendations_1.extend(user1_final_content_recs) # Add the movies to the final recs list
    else:
        user1_final_content_recs = [] # Create an empty list
        for movie in unseen_user1_recommended_content_1_recs: # Run through each movie
            if movie not in final_recommendations_1: # IF the movie is not already in the final recs list
                user1_final_content_recs.append(movie)  # Add them to the list
        final_recommendations_1.extend(user1_final_content_recs)  # Add the content recs to the final recs list
else:
    user1_top_movie_content = content[content['title'] == user1_top_movie].iloc[0, 1:11].tolist() # Get the 10 most similar movies to the user's top rated movie
    # Remove movies that user has already seen.
    unseen_user1_top_movie_content_recs = [x for x in user1_top_movie_content if x not in user_1_movies_watched]
    # Return First 5 movies in the unseen movie list
    if len(unseen_user1_top_movie_content_recs) >= 5:
        user1_top_movie_5_content_recs = []  # Create an empty list to store the top 5 content based movies
        for movie in unseen_user1_top_movie_content_recs:   # Run through each of the unseen content based recs
            if movie not in final_recommendations_1:  # If the movie is not already in the recommendation list
                user1_top_movie_5_content_recs.append(movie)  # Add it to the list
            if len(user1_top_movie_5_content_recs) == 5:  # Stop when we have 5 suggestions
                break
        final_recommendations_1.extend(user1_top_movie_5_content_recs)  # Append suggestions to the final list
    else:
        user1_top_movie_5_content_recs = []  # Create an empty list to store the top 5 content based movies
        for movie in unseen_user1_top_movie_content_recs:  # Run through each of the unseen content based recs
            if movie not in final_recommendations_1:  # If the movie is not already recommended
                user1_top_movie_5_content_recs.append(movie)  # Add the movie to the list
        final_recommendations_1.extend(user1_top_movie_5_content_recs)  # Add recs to the final list

# View the List after Step 4
print(final_recommendations_1)

# Step 5 Fill The rest with High Rated/Popular movies
# If the recommended_movies is not at 10, then we will add the most popular/highest rated movie to the list until it is
if len(final_recommendations_1) < 10:
    popular_movies = binned_movie_ratings.sort_values(by=['Popularity Score', 'Average Rating'], ascending=False)['title'].tolist()
    for movie in popular_movies:
        if movie not in user_1_movies_watched or final_recommendations_1:
            final_recommendations_1.extend(movie)
            if len(final_recommendations_1) == 10:
                break


# Print the final recommendations list
print(final_recommendations_1) # Print the final list
len(final_recommendations_1) # Check to see if Length is 10





# Apply the above code in the form of a 'for' loop
final_recommendations_df = pd.DataFrame(columns=['userID', 'recommendations'])
# Create the for loop
for userID in collab['userID'].unique():
    final_recommendations = []
    # Step 1 Create a List of all the movies each User has watched
    # Create a list for each user
    user_movies_grouped = user_movies.groupby('userID')['movie_watched'].agg(list).reset_index()
    # Final List of every movie user 1 has watched
    user_x_movies_watched = user_movies_grouped.loc[user_movies_grouped['userID'] == userID, 'movie_watched'].values[0]
    # Step 2 get the user's highest rated movie
    user_x_top_movie = top_rated_movie[top_rated_movie['userID'] == userID].iloc[0]['title']
    # Step 3 Get the first 5 names in the collab list (Already filtered to make sure the user has not seen before)
    collab_recommended_movies_x = ast.literal_eval(collab.loc[collab['userID'] == userID, 'recommended_movies'].values[0]) # Go to the recommended movies column for the user we are looking at
    collab_recommended_movies_x_recs = collab_recommended_movies_x[:5]  # Return the first 5 values in the list
    # Add the Collab Values to the List
    final_recommendations.extend(collab_recommended_movies_x_recs)
    # Step 4 Check to see if there are any movies in the above list, if not, use the user's top_rated_movie to suggest
    if len(final_recommendations) >= 1:
        # Find the movie out of the suggested movies that has the highest rating and popularity score
        # Select the Rows in the Binned Ratings Table for each movie selected in the collab
        filtered_df_x = binned_movie_ratings.loc[binned_movie_ratings['title'].isin(final_recommendations)]
        # Find the movie out of the suggested with the highest Popularity Score and Average Rating
        top_movie_out_of_collab = filtered_df_x.sort_values(by=['Popularity Score', 'Average Rating'], ascending=False).iloc[0]['title']
        # Return the Top 5 most similar movies to the 'top_movie'
        full_recommended_content_x = content.loc[content['title'] == top_movie_out_of_collab, :].values.tolist()[0][1:11]
        # Remove movies that user has already seen.
        unseen_user_x_collab_based_content_recs = [x for x in full_recommended_content_x if x not in user_x_movies_watched]
        # Return First 5 movies in the unseen movie list
        if len(unseen_user_x_collab_based_content_recs) >= 5:
            user_x_final_content_recs = []  # Create an empty list
            for movie in unseen_user_x_collab_based_content_recs:  # Run through each of the movies
                if movie not in final_recommendations:  # If the movie is not already recommended
                    user_x_final_content_recs.append(movie)  # Add it to the above list
                if len(user_x_final_content_recs) == 5:  # Add at most 5 movies
                    break
            final_recommendations.extend(user_x_final_content_recs)  # Add the movies to the final recs list
        else:
            user_x_final_content_recs = []  # Create an empty list
            for movie in unseen_user_x_collab_based_content_recs:  # Run through each movie
                if movie not in final_recommendations:  # IF the movie is not already in the final recs list
                    user_x_final_content_recs.append(movie)  # Add them to the list
            final_recommendations.extend(user_x_final_content_recs)  # Add the content recs to the final recs list
    else:
        user_x_top_movie_content = content[content['title'] == user_x_top_movie].iloc[0, 1:11].tolist() # Get the 10 most similar movies to the user's top rated movie
        # Remove movies that user has already seen.
        unseen_user_x_top_movie_based_content_recs = [x for x in user_x_top_movie_content if x not in user_x_movies_watched]
        # Return First 5 movies in the unseen movie list
        if len(unseen_user_x_top_movie_based_content_recs) >= 5:
            user_x_top_movie_5_content_recs = []  # Create an empty list to store the top 5 content based movies
            for movie in unseen_user_x_top_movie_based_content_recs:  # Run through each of the unseen content based recs
                if movie not in final_recommendations:  # If the movie is not already in the recommendation list
                    user_x_top_movie_5_content_recs.append(movie)  # Add it to the list
                if len(user_x_top_movie_5_content_recs) == 5:  # Stop when we have 5 suggestions
                    break
            final_recommendations.extend(user_x_top_movie_5_content_recs)  # Append suggestions to the final list
        else:
            user_x_top_movie_5_content_recs = []  # Create an empty list to store the top 5 content based movies
            for movie in unseen_user_x_top_movie_based_content_recs:  # Run through each of the unseen content based recs
                if movie not in final_recommendations:  # If the movie is not already recommended
                    user_x_top_movie_5_content_recs.append(movie)  # Add the movie to the list
            final_recommendations.extend(user_x_top_movie_5_content_recs)  # Add recs to the final list
    # Step 5 Fill The rest with High Rated/Popular movies
    # If the recommended_movies is not at 10, then we will add the most popular/highest rated movie to the list until it is
    if len(final_recommendations) < 10:
        popular_movies = binned_movie_ratings.sort_values(by=['Popularity Score', 'Average Rating'], ascending=False)['title'].tolist()
        for movie in popular_movies:
            if movie not in user_x_movies_watched and movie not in final_recommendations:
                final_recommendations.append(movie)
                if len(final_recommendations) == 10:
                    break
    final_recommendations_df = final_recommendations_df.append({'userID': userID, 'recommendations': final_recommendations}, ignore_index=True)

# Print the final recommendations DataFrame
print(final_recommendations_df)

# Export the Recommendation dataframe to a csv
final_recommendations_df.to_csv('final_hybrid_recs.csv')
