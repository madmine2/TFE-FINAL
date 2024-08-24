import requests
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access environment variables
api_key = os.getenv("CHATGPT_3_5_TURBO_API_KEY")
model = "gpt-3.5-turbo"
# model = "gpt-4"



def apiCall(instruction, temp,seed = None, max_retries=3):
    if api_key is None:
        raise ValueError("OpenAI API key is not set in environment variables.")

    client = OpenAI(
    api_key=api_key,
    )

    chat_completion = client.chat.completions.create(
        messages=[
                 {"role": "system", 
                  "content": """ 
You are a top expert at verifying descriptions based on fact and completing them if a fact is missing.
You are given a description and a list of facts that should be in the description.
If a fact from the list is missing from the description then you complete the description to include that fact.
At the end all facts must be included in the description.
You only output the final description and you are rewarded with a lot of money if there is no missing fact.

 """
    },
    {"role": "user","content": instruction },
        ],
        model=model,
        temperature= temp,
        seed = seed,
    )
    response = chat_completion.choices[0].message.content
    return response
    

if __name__ == "__main__":
    # Example usage
    print(apiCall("example text", "example context",123,0.5))
    
    """
    You are a helpfull assistant, expert at making descriptions of movies based on informations provided.
You are given a list of informations about a show.
Write a small description of this movie using those informations.
You must absolutely include all the informations given to you in your description of the movie.
Do not forget to include duration and content rating.
    Here are a 2 examples to refere to : 
    
<user> : movie type = Movie
title = Tremors 2: Aftershocks
director = S.S. Wilson
cast = Fred Ward, Chris Gartin, Helen Shaver, Michael Gross, Marcelo Tubert, Marco Hernandez, José Ramón Rosario, Thomas Rosales Jr.
date added to Netflix = January 1, 2020
release year = 1995
content rating = PG-13
<assistant> : "Tremors 2: Aftershocks" is a PG-13 movie directed by S.S. Wilson that was released in 1995. The film features a cast including Fred Ward, Chris Gartin,Marcelo Tubert, Marco Hernandez, José Ramón Rosario, Thomas Rosales Jr, Helen Shaver and Michael Gross. It was added to streaming platforms on January 1, 2020. The movie is an action-packed sequel that follows a group of individuals as they face off against deadly underground creatures known as Graboids. This film promises suspense, humor, and thrilling escapades as the characters try to survive against these monstrous foes.
    
<user> :   movie type = Movie
title = Fat Ballerina - David A. Arnold
director = Milton Horowitz
cast = David A. Arnold
date added to Netflix = March 2, 2020
release year = 2020
content rating = TV-MA
duration = 61 min

<assistant> : "Fat Ballerina" is a humorous movie directed by Milton Horowitz and starring David A. Arnold. Released in 2020, the film has a TV-MA rating and a duration of 61 minutes. The story follows the comedic journey of a character named Fat Ballerina, portrayed by David A. Arnold, as he navigates life's challenges with a touch of humor and wit. This movie added in March 2, 2020 to the streaming plateform promises an entertaining and light-hearted experience for viewers.
   """