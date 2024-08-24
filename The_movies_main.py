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




numberOfMovie = 0
numberOfFactsCategories = 0
numberOfGoodFactsCategories = 0
GPTTemp = 0
GPTTempConfirmation = 0
def structured_description():
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
        for j in randomizedRange:
            tempFact = str(movieDf2[column_names[j]][i])
            tempFact = column_names[j] + " = " + tempFact + " \n"
            facts.append(tempFact)
            numberOfFacts1 +=1
            if t < numberOfGoodFactsCategories : 
                instruction = instruction + tempFact
                t+= 1
           
            
            
        result = apiCallGpt(instruction,GPTTemp)
        instructionConfirmation = instruction + "\n"+ result
        if i == 0:
            print(instructionConfirmation)
        resultFinal = apiCallGptConfirmation(instructionConfirmation,GPTTempConfirmation)
        if i ==0 : 
            print(resultFinal)
        listOfDescriptions.append([resultFinal, facts, instruction, numberOfGoodFactsCategories+2])
        
        
    #Display a number (x) of random descriptions with the informations that were given for generating it
    # x = min(5, numberOfMovie)
    # random_numbers = random.sample(range(0, numberOfMovie), x)
    # for i in random_numbers: 
    #     print("Instructions : "+ listOfDescriptions[i][2])
    #     print("Structured description : "+ listOfDescriptions[i][0] + "\n")
    return listOfDescriptions





# ollamaTemp = 0
# model = 0
# def identify_facts(listOfDescriptions):
#     # Request to identifie facts and accuracy, given a context, a description and a list of facts about a movie for numberOfMovie
#     listOfFoundFacts = []
#     numberOfFacts2 = 0
#     numberOfRightFacts = 0
#     numberOfbadFacts = 0
#     scoreFalseNegative = 0
#     scoreTruePositive = 0
#     scoreFalsePositive = 0
#     scoreTrueNegative = 0
#     scoreBadOutput = 0
#     for i in range (numberOfMovie):
#         if i % 10 == 0:
#             print(i)
#         instruction = ""
#         listOfFoundFacts.append([])
#         numberOfGoodFactsCategories = listOfDescriptions[i][3]
#         t00 = time.time()
#         for j in range(2,len(listOfDescriptions[i][1])):
#             numberOfFacts2 += 1
#             instruction = "Answer correct if you find the fact in the description and wrong if you don't \n The fact is : [" + listOfDescriptions[i][1][j]+"] and the description is [" + listOfDescriptions[i][0] + "]"
#             t0 = time.time()
#             result = apiCallOllama(instruction,ollamaTemp, model).lower()
#             t1 = time.time()
#             #print(f"One api call done in {t1-t0} seconds")
#             if j >= numberOfGoodFactsCategories :
#                 # the facts after numberOfGoodFacts should not be in the description
#                 numberOfbadFacts += 1 
#                 if "correct" in result: 
#                     scoreFalsePositive +=1
#                     listOfFoundFacts[i].append( [listOfDescriptions[i][1][j], listOfDescriptions[i][0], result, "FP  -  FalsePositive"])
#                 elif "wrong" in result: 
#                     scoreTrueNegative += 1
#                     listOfFoundFacts[i].append( [listOfDescriptions[i][1][j], listOfDescriptions[i][0], result, "TN  -  TrueNegative"])
#                 else : 
#                     scoreBadOutput += 1
#                     listOfFoundFacts[i].append( [listOfDescriptions[i][1][j], listOfDescriptions[i][0], result, "!!! BAD OUTPUT !!!"])
#             else :
#                 # every fact here is supposed to be found
#                 numberOfRightFacts +=1
#                 if "correct" in result: 
#                     scoreTruePositive +=1
#                     listOfFoundFacts[i].append( [listOfDescriptions[i][1][j], listOfDescriptions[i][0], result, "TP  -  TruePositive"])
#                 elif "wrong" in result: 
#                     scoreFalseNegative += 1
#                     listOfFoundFacts[i].append( [listOfDescriptions[i][1][j], listOfDescriptions[i][0], result, "FN  -  FalseNegative"])
#                 else : 
#                     scoreBadOutput += 1
#                     listOfFoundFacts[i].append( [listOfDescriptions[i][1][j], listOfDescriptions[i][0], result, "!!! BAD OUTPUT !!!"])
#         t11 = time.time()
#         #print(f"One movie done in {t11-t00} seconds")

#     # make the metrics                
                    
#     Accuracy = (scoreTrueNegative + scoreTruePositive) / numberOfFacts2 * 100
#     precision = scoreTruePositive / (scoreFalsePositive + scoreTruePositive ) * 100
#     recall = scoreTruePositive / numberOfRightFacts * 100
#     specificity  = scoreTrueNegative / numberOfbadFacts * 100
#     falsePositiveRate = scoreFalsePositive / (scoreFalsePositive + scoreTrueNegative ) *100
#     fMeasure = 2 * precision * recall /( precision + recall)


#     # write all result in a txt file
#     result_file_number = int(open("LLms/fileNumber.txt","r").readline())
#     result_file_number += 1
#     open("LLms/fileNumber.txt","w").writelines(str(result_file_number))
#     fileName = "results/result" + str(result_file_number)+".txt"
#     with open(fileName,"w") as fout:
#         fout.writelines("techniques  = basics \n")
#         fout.writelines("numberOfAgent = 1 \n")
#         fout.writelines("Model =  " + model + "\n")
#         fout.writelines("GPT temp =  " + str(GPTTemp) + "\n")
#         fout.writelines("ollama temp =  " + str(ollamaTemp) + "\n")
#         fout.writelines("Accuracy =  " + str(Accuracy) + "\n")
#         fout.writelines("precision =  " + str(precision) + "\n")
#         fout.writelines("recall =  " + str(recall) + "\n")
#         fout.writelines("specificity =  " + str(specificity) + "\n")
#         fout.writelines("falsePositiveRate =  " + str(falsePositiveRate) + "\n")
#         fout.writelines("fMeasure =  " + str(fMeasure) + "\n \n \n")
#         fout.writelines("NumberOfMovies  = "+ str(numberOfMovie) + "\n")
#         fout.writelines("NumberOfFacts  = "+ str(numberOfFacts2) + "\n")
#         fout.writelines("numberOfFactsCategories  = "+ str(numberOfFactsCategories) + "\n")
#         fout.writelines("numberOfGoodFactsCategories  = "+ str(numberOfGoodFactsCategories) + "\n")
#         fout.writelines("numberOfRightFacts =  " + str(numberOfRightFacts) + " and numberOfbadFacts  = "+ str(numberOfbadFacts) + "\n \n \n")
#         fout.writelines("scoreBadOutput =  " + str(scoreBadOutput) + "\n")
#         fout.writelines("scoreTruePositive =  " + str(scoreTruePositive) + "\n")
#         fout.writelines("scoreTrueNegative =  " + str(scoreTrueNegative) + "\n")
#         fout.writelines("scoreFalsePositive =  " + str(scoreFalsePositive) + "\n")
#         fout.writelines("scoreFalseNegative =  " + str(scoreFalseNegative) + "\n \n")
#         for i in range(numberOfMovie):
#             try :
#                 fout.writelines(listOfDescriptions[i][2]+"\n")
#             except UnicodeEncodeError as e:
#                 pass
            
#             for j in range(len(listOfFoundFacts[i])):
#                 try :
#                     fout.writelines(listOfFoundFacts[i][j][3]+"\n")
#                 except UnicodeEncodeError as e:
#                     pass
#                 try :
#                     fout.writelines(listOfFoundFacts[i][j][2]+"\n")
#                 except UnicodeEncodeError as e:
#                     pass
#                 try :
#                     fout.writelines("fact : "+ listOfFoundFacts[i][j][0]+"\n")
#                 except UnicodeEncodeError as e:
#                     pass
#                 try : 
#                     fout.writelines( listOfFoundFacts[i][j][1]+" \n \n \n")
#                 except UnicodeEncodeError as e:
#                     pass
#             fout.writelines("\n \n")






ollamaTemp = 0
model = 0
def identify_facts_parallel_agents(listOfDescriptions, numberOfAgent):
    # Request to identifie facts and accuracy, given a context, a description and a list of facts about a movie for numberOfMovie
    listOfFoundFacts = []
    numberOfFacts2 = 0
    numberOfRightFacts = 0
    numberOfbadFacts = 0
    scoreFalseNegative = 0
    scoreTruePositive = 0
    scoreFalsePositive = 0
    scoreTrueNegative = 0
    scoreBadOutput = 0
    for i in range (numberOfMovie):
        if i % 50 == 0:
            print(i)
        instruction = ""
        listOfFoundFacts.append([])
        numberOfGoodFactsCategories = listOfDescriptions[i][3]
        for j in range(2,len(listOfDescriptions[i][1])):
            result = []
            numberOfFacts2 += 1
            instruction = "The fact is : [" + listOfDescriptions[i][1][j]+"] and the description is [" + listOfDescriptions[i][0] + "]"
            for agent in range(numberOfAgent):
                result.append(apiCallOllama(instruction,ollamaTemp, model).lower())
                
                
            numberOfCorrect = 0
            numberOfWrong = 0
            for k in result:
                if "correct" in k:
                    if not "wrong" in k and not "incorrect" in k:
                        numberOfCorrect +=1
                elif "wrong" in k:
                    if not re.search(r'\bcorrect\b', k):
                        numberOfWrong +=1
                else :
                    pass
            if j < numberOfGoodFactsCategories :
               # every fact here is supposed to be found
                numberOfRightFacts +=1
                if numberOfCorrect > numberOfWrong and numberOfCorrect > 0: 
                    scoreTruePositive +=1
                    listOfFoundFacts[i].append( [listOfDescriptions[i][1][j], listOfDescriptions[i][0], result, "TP  -  TruePositive"])
                elif numberOfCorrect < numberOfWrong: 
                    scoreFalseNegative += 1
                    listOfFoundFacts[i].append( [listOfDescriptions[i][1][j], listOfDescriptions[i][0], result, "FN  -  FalseNegative"])
                else : 
                    scoreBadOutput += 1
                    listOfFoundFacts[i].append( [listOfDescriptions[i][1][j], listOfDescriptions[i][0], result, "!!! BAD OUTPUT !!!"])
            else :
                 # the facts after numberOfGoodFacts should not be in the description
                numberOfbadFacts += 1 
                if numberOfCorrect > numberOfWrong and numberOfCorrect > 0: 
                    scoreFalsePositive +=1
                    listOfFoundFacts[i].append( [listOfDescriptions[i][1][j], listOfDescriptions[i][0], result, "FP  -  FalsePositive"])
                elif numberOfCorrect < numberOfWrong: 
                    scoreTrueNegative += 1
                    listOfFoundFacts[i].append( [listOfDescriptions[i][1][j], listOfDescriptions[i][0], result, "TN  -  TrueNegative"])
                else : 
                    scoreBadOutput += 1
                    listOfFoundFacts[i].append( [listOfDescriptions[i][1][j], listOfDescriptions[i][0], result, "!!! BAD OUTPUT !!!"])

    
    # make the metrics                
                    
    Accuracy = (scoreTrueNegative + scoreTruePositive) / numberOfFacts2 * 100
    precision = scoreTruePositive / (scoreFalsePositive + scoreTruePositive ) * 100
    recall = scoreTruePositive / numberOfRightFacts * 100
    specificity  = scoreTrueNegative / numberOfbadFacts * 100
    falsePositiveRate = scoreFalsePositive /numberOfbadFacts *100
    fMeasure = 2 * precision * recall /( precision + recall)


    # write all result in a txt file
    result_file_number = int(open("LLms/fileNumber.txt","r").readline())
    result_file_number += 1
    open("LLms/fileNumber.txt","w").writelines(str(result_file_number))
    fileName = "results/result" + str(result_file_number)+".txt"
    with open(fileName,"w") as fout:
        fout.writelines("techniques  = basics \n")
        fout.writelines("numberOfAgent = "+str(numberOfAgent)+ "\n")
        fout.writelines("Model =  " + model + "\n")
        fout.writelines("GPT temp =  " + str(GPTTemp) + "\n")
        fout.writelines("ollama temp =  " + str(ollamaTemp) + "\n")
        fout.writelines("Accuracy =  " + str(Accuracy) + "\n")
        fout.writelines("precision =  " + str(precision) + "\n")
        fout.writelines("recall =  " + str(recall) + "\n")
        fout.writelines("specificity =  " + str(specificity) + "\n")
        fout.writelines("falsePositiveRate =  " + str(falsePositiveRate) + "\n")
        fout.writelines("fMeasure =  " + str(fMeasure) + "\n \n \n")
        fout.writelines("NumberOfMovies  = "+ str(numberOfMovie) + "\n")
        fout.writelines("NumberOfFacts  = "+ str(numberOfFacts2) + "\n")
        fout.writelines("numberOfFactsCategories  = "+ str(numberOfFactsCategories) + "\n")
        fout.writelines("numberOfGoodFactsCategories  = "+ str(numberOfGoodFactsCategories) + "\n")
        fout.writelines("numberOfRightFacts =  " + str(numberOfRightFacts) + " and numberOfbadFacts  = "+ str(numberOfbadFacts) + "\n \n \n")
        fout.writelines("scoreBadOutput =  " + str(scoreBadOutput) + "\n")
        fout.writelines("scoreTruePositive =  " + str(scoreTruePositive) + "\n")
        fout.writelines("scoreTrueNegative =  " + str(scoreTrueNegative) + "\n")
        fout.writelines("scoreFalsePositive =  " + str(scoreFalsePositive) + "\n")
        fout.writelines("scoreFalseNegative =  " + str(scoreFalseNegative) + "\n \n")
        for i in range(numberOfMovie):
            try :
                fout.writelines(listOfDescriptions[i][2]+"\n")
            except UnicodeEncodeError as e:
                pass
            
            for j in range(len(listOfFoundFacts[i])):
                try :
                    fout.writelines(listOfFoundFacts[i][j][3]+"\n")
                except UnicodeEncodeError as e:
                    pass
                try :
                    fout.writelines(str(listOfFoundFacts[i][j][2])+"\n")
                except UnicodeEncodeError as e:
                    pass
                try :
                    fout.writelines("fact : "+ listOfFoundFacts[i][j][0]+"\n")
                except UnicodeEncodeError as e:
                    pass
                try : 
                    fout.writelines( listOfFoundFacts[i][j][1]+" \n \n \n")
                except UnicodeEncodeError as e:
                    pass
            fout.writelines("\n \n")



T00 = time.time()
# "llama3_fact","llama3_fact4","lamma3_fact_justification2","llama3_fine_tune","llama3_just_fineTune_05","llama3_just_fineTune_2","llama3_just_fineTune_1"
#"llama3OutputFormat","llama3FewShot","llama3Thought1","llama3Thought2","llama3FewShot1","llama3OutputFormat2"
models = ["llama3Interaction_E"]
GPTTemp = 0.2
ollamaTemp = 1
numberOfMovie = 300


listOfDescriptions = structured_description()


iter = 1
for k in models :
    model = k
    numberOfAgent = 1
    t0 = time.time()
    #identify_facts(listOfDescriptions)
    identify_facts_parallel_agents(listOfDescriptions,numberOfAgent)
    Time = time.time()-t0
    print("model "+str(iter) +" is done, it took " + str(Time) + " seconds")
    iter += 1

model = "llama3Interaction_A"
numberOfAgent = 3
t0 = time.time()
ollamaTemp = 1
identify_facts_parallel_agents(listOfDescriptions,numberOfAgent)
Time = time.time()-t0
print("iteration 4 is done, it took " + str(Time) + " seconds")

T11 = time.time()-T00
T11 = int(T11/60)
print("L'algo a pris "+str(T11)+" minutes au total" )
   