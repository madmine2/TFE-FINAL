from LLms.ollamaLllm import apiCall as apiCallOllama
from LLms.gptLLm import apiCall as apiCallGpt
from LLms.gptLLMConfirmation import apiCall as apiCallGptConfirmation
import random
import pandas as pd
import numpy as np
import os
import time
import ast
import langcodes
import datetime
import re


# Load and merge datasets
file_path1 = 'Datasets/the movies dataset/credits.csv'
file_path2 = 'Datasets/the movies dataset/movies_metadata.csv'
movieDf1 = pd.read_csv(file_path1)
movieDf2 = pd.read_csv(file_path2)

movieDf1['id'] = movieDf1['id'].astype(str)
movieDf2['id'] = movieDf2['id'].astype(str)
movieDf2 = movieDf2.merge(movieDf1, on='id')

# Remove unnecessary columns
columns_to_drop = ['genres', 'adult', 'belongs_to_collection', 'homepage', 'imdb_id', 'id', 
                   'poster_path', 'spoken_languages', 'status', 'original_title', 'video', 'vote_count']
movieDf2.drop(columns_to_drop, axis=1, inplace=True)

# Function to generate random budget/revenue values
def generate_random_value():
    return np.random.uniform(500000, 300000000)

# Replace invalid budget and revenue values with random values
movieDf2['budget'] = movieDf2['budget'].apply(lambda x: generate_random_value() if pd.isna(x) or x < 3 else x)
movieDf2['revenue'] = movieDf2['revenue'].apply(lambda x: generate_random_value() if pd.isna(x) or x < 3 else x)

# Clean up the dataset
movieDf2 = movieDf2.dropna()
# Convert datatypes and round values
movieDf2['budget'] = movieDf2['budget'].astype(int)
movieDf2['popularity'] = movieDf2['popularity'].astype(float)
movieDf2['revenue'] = movieDf2['revenue'].astype(int)
movieDf2['runtime'] = movieDf2['runtime'].astype(int)
movieDf2['vote_average'] = movieDf2['vote_average'].astype(float)
movieDf2 = movieDf2[movieDf2['runtime'] >= 3.0]
movieDf2 = movieDf2.round({'popularity': 1, 'vote_average': 1})
movieDf2.drop('popularity', axis=1, inplace=True)

# Helper functions for data extraction and formatting
def extract_first_name(list_str):
    """
    Extracts the 'name' from the first dictionary in a string representation of a list.
    Used for production companies and countries.
    """
    try:
        parsed_list = ast.literal_eval(list_str)
        if isinstance(parsed_list, list) and len(parsed_list) > 0:
            if isinstance(parsed_list[0], dict) and 'name' in parsed_list[0]:
                return parsed_list[0]['name']
        return None
    except (ValueError, SyntaxError):
        return None

def iso_code_to_language(code):
    """
    Converts ISO language code to full language name.
    Example: 'en' -> 'English'
    """
    return langcodes.Language.get(code).display_name() 

def extract_director_producer(crew_list_str):
    """
    Extracts the names of the director and producer from a crew list string.
    """
    crew_list = ast.literal_eval(crew_list_str)
    director = next((member['name'] for member in crew_list if member['job'] == 'Director'), None)
    producer = next((member['name'] for member in crew_list if member['job'] == 'Producer'), None)
    return director, producer

def convert_and_extract_actors_names(list_str):
    """
    Extracts up to four actor names from a string representation of a cast list.
    """
    try:
        list = ast.literal_eval(list_str)[:4]
        return ', '.join([value['name'] for value in list])
    except (ValueError, SyntaxError):
        return ''

def budget_to_millions(number):
    """
    Converts a number to millions of dollars format.
    Example: 150000000 -> "150.0 millions of dollars"
    """
    millions = number / 1_000_000
    return f"{millions:.1f} millions of dollars"

def convert_minutes_to_hours(minutes):
    """
    Converts minutes to hours and minutes format.
    Example: 142 -> "2h22"
    """
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}"

def date_to_sentence(date_str):
    """
    Convert a date string from the format yyyy-mm-dd to a readable sentence.
    """
    date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    
    day = date_obj.day
    month = date_obj.strftime("%B")  # Get full month name
    year = date_obj.year
    
    sentence = f"the {day} of {month} {year}"
    return sentence

# Apply data transformations
movieDf2['production_companies'] = movieDf2['production_companies'].apply(extract_first_name)
movieDf2['production_countries'] = movieDf2['production_countries'].apply(extract_first_name)
movieDf2['original_language'] = movieDf2['original_language'].apply(iso_code_to_language)
movieDf2[['Director', 'Producer']] = movieDf2['crew'].apply(lambda x: pd.Series(extract_director_producer(x)))
movieDf2 = movieDf2.dropna()
movieDf2['cast'] = movieDf2['cast'].apply(convert_and_extract_actors_names)
movieDf2['budget'] = movieDf2['budget'].apply(budget_to_millions)
movieDf2['revenue'] = movieDf2['revenue'].apply(budget_to_millions)
movieDf2['runtime'] = movieDf2['runtime'].apply(convert_minutes_to_hours)
movieDf2['release_date'] = movieDf2['release_date'].apply(date_to_sentence)
movieDf2.drop('crew', axis=1, inplace=True)

# Rename columns for clarity
column_renames = {
    'original_language': 'original language of the movie',
    'production_companies': 'production compagnie',
    'production_countries': 'production countrie',
    'release_date': 'date of release',
    'vote_average': 'average note',
    'runtime': 'duration of the movie',
}
movieDf2.rename(columns=column_renames, inplace=True)

# Remove rows with invalid Producer or Director
movieDf2 = movieDf2[movieDf2['Producer'].notna() & (movieDf2['Producer'].str.strip() != 'None')]
movieDf2 = movieDf2[movieDf2['Director'].notna() & (movieDf2['Director'].str.strip() != 'None')]

# Reorder columns
desired_order = ['title', 'overview', 'budget', 'original language of the movie', 'production compagnie', 'production countrie', 
                 'date of release', 'revenue', 'duration of the movie', 'tagline', 'average note', 'cast', 'Director', 'Producer']
movieDf2 = movieDf2[desired_order]

# Display results
print("preprocess done")




numberOfMovie = 300
numberOfFactsCategories = 0
numberOfGoodFactsCategories = 0
GPTTemp = 0.2
GPTTempConfirmation = 0

# Request to generate the structured descriptions, given a context and a list of facts about a movie for numberOfMovie
column_names = movieDf2.columns
numberOfFactsCategories = len(column_names)
listOfDescriptions = []
random_sampled_df = movieDf2.sample(n=numberOfMovie)
random_sampled_df = random_sampled_df.index.tolist()
# We make a list with every fact and a text with every fact except the last one. This text will be used as a prompt. Also if a fact include "nan", we don't use it
# every thing is stored in listOfDescriptions with a length of numberOfMovie
# listOfDescriptions[i][0] is the structured description of the movie i 
# listOfDescriptions[i][1] is the list of facts
# listOfDescriptions[i][2] is the text with the facts except the last one
numberOfFacts1 = 0
for i in random_sampled_df:
    if i == 0:
        continue
    instruction = ""
    facts = []
    randomizedRange= random.sample(range(2, numberOfFactsCategories), numberOfFactsCategories-2)
    
    numberOfGoodFactsCategories =  random.sample(range(len(randomizedRange)+1), 1)[0]
    t = 0 # track the number of "good" facts (facts that will be given to the llm), when t is more than numberOfGoodFacts, the facts won't be given to the llm
    for p in range(2):
        tempFact = str(movieDf2[column_names[p]][i])
        tempFact = column_names[p] + " = " + tempFact + " \n"
        facts.append(tempFact)
        numberOfFacts1 +=1
        instruction = instruction + tempFact
    tempValues = []
    for j in randomizedRange:
        tempFact = str(movieDf2[column_names[j]][i])
        tempFact2 = column_names[j] + " = " + tempFact + " \n"
        facts.append(tempFact2)
        numberOfFacts1 +=1
        if t < numberOfGoodFactsCategories : 
            instruction = instruction + tempFact2
            t+= 1
            tempValues.append(["correct",column_names[j] + " = " + tempFact])
        else:
            tempValues.append(["wrong",column_names[j] + " = " + tempFact])
       
        
        
    result = apiCallGpt(instruction,GPTTemp)
    instructionConfirmation = instruction + "\n"+ result
    resultFinal = apiCallGptConfirmation(instructionConfirmation,GPTTempConfirmation)
    for tempValue in tempValues:
        tempValue.append(resultFinal)
        listOfDescriptions.append([tempValue])
    
with open("dataForFinetune.txt","w") as fout:
    for descr in listOfDescriptions:
        for tempValue in descr:
            for value in tempValue:
                fout.writelines(value+"\n&&&\n")
            fout.writelines("\n@@\n")
#Display a number (x) of random descriptions with the informations that were given for generating it
# x = min(5, numberOfMovie)
# random_numbers = random.sample(range(0, numberOfMovie), x)
# for i in random_numbers: 
#     print("Instructions : "+ listOfDescriptions[i][2])
#     print("Structured description : "+ listOfDescriptions[i][0] + "\n")

