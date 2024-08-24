from ollamaLllm import apiCall as apiCallOllama
from gptLLm import apiCall as apiCallGpt
import random

instruction = "You are a medical expert, answer questions using facts an accuracy"
context = "In one line, can you explain what is alzheimer please"
# instruction = "You are a helpful code assistant. Your task is to generate a valid JSON object based on the given information:"
# context = """name: John
# lastname: Smith
# address: #1 Samuel St.
# Just generate the JSON object without explanations:"""
seed = random.randint(1, 1000)
temp = 0.5
result = apiCallOllama(instruction, context,seed,temp)
print(seed)
print(result)
result = apiCallGpt(instruction, context,seed,temp)
print(result)