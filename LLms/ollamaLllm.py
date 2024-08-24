import requests
import time
import re

url = "http://localhost:11434/api/generate"


def apiCall(instruction, temp, model, seed = None,  max_retries=3):
    # Define the data payload as a dictionary
    data = {
        "model": model,
        "prompt" :   instruction,
        "stream": False,
        "options": {
            "seed": seed,
            "temperature": temp
         }
    }
    
    # Set the timeout duration to 2 minutes (in seconds)
    timeout_duration = 120
    
    # Counter for the number of retries
    retries = 0
    while retries < max_retries:
        try:
            # Send a POST request with the data payload and timeout
            result = requests.post(url, json=data, timeout=timeout_duration)
            
            # Check if the request was successful (status code 200)
            if result.status_code == 200:
                result = result.json()["response"]
                result = result.replace("\n","")
                result = result.replace("  ","")
                return result
            else:
                print(f"Request failed with status code {result.status_code}")
                print(result)
                return "error"
        except requests.exceptions.Timeout:
            retries += 1
            print(f"Request timed out after {timeout_duration} seconds. Retrying... (Attempt {retries}/{max_retries})")
            time.sleep(1)  # Optional delay between retries
            
    print(f"Failed after {max_retries} retries. Giving up.")
    return "timeout"

if __name__ == "__main__":

    # instruction ="Here is the categorie : 'movie type' and here is the fact in that categorie : 'TV show'. \n And here is the description : Numberblocks is a British TV show that was added on September 15, 2021. This educational show is targeted towards young children as it has a TV-Y rating. The show consists of 6 seasons and features a cast including Beth Chalmers, David Holt, Marcel McCalla, and Teresa Gallagher. Numberblocks likely focuses on teaching children basic math concepts in a fun and engaging way, using colorful characters and entertaining stories to make learning numbers enjoyable." 
    # # instruction = "You are a helpful code assistant. Your task is to generate a valid JSON object based on the given information:"
    # # context = """name: John
    # # lastname: Smith
    # # address: #1 Samuel St.
    # # Just generate the JSON object without explanations:"""
    # temp = 1
    # for i in range(1): 
    #     result = apiCall(instruction,temp, 'gemma_fact')
    #     print(result)
    pass
    
