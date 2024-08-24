from classToReadModelsOutput import ModelMetrics
import matplotlib.pyplot as plt
import os

def importFiles(folder_path,modelName = None,numberOfMovies = None,GPTtemp = None,numberOfGoodFactsCategories = None,ollamatemp = None, techniques = None):
    # Construct the criteria list
    criteria = [
        ("techniques", techniques),
        ("Model", modelName),
        ("NumberOfMovies", numberOfMovies),
        ("GPTtemp", GPTtemp),
        ("numberOfGoodFactsCategories", numberOfGoodFactsCategories),
        ("ollamatemp", ollamatemp)
    ]
    
    # Filter out None values
    criteria = [(key, value) for key, value in criteria if value is not None]
    models = []
    
    # List all files in the folder
    file_names = os.listdir(folder_path)
    # Iterate over each file
    for file_name in file_names:
        if file_name.endswith(".txt"):
            # Construct the file path
            file_path = os.path.join(folder_path, file_name)
            # Create an instance of ModelMetrics and append to models list
            models.append(ModelMetrics(file_path))
    
    # Filter the list of models based on the criteria
    filtered_models = []
    for model in models:
        # Check if all criteria are satisfied
        modelPass = True
        for attr, value in criteria:
            if isinstance(value, list):
                # print(f'attr : {getattr(model, attr)}')
                # print(f'value : {value}')
                if  getattr(model, attr) in value:
                    pass
                else :
                    # print("modelPass = false")
                    modelPass = False
            else:
                if not getattr(model, attr) == value:
                    modelPass = False
        if modelPass == True:
            filtered_models.append(model)
    # print(len(filtered_models))
    # print(len(models))
    return filtered_models


def makeGraph(attrXAxis, attrYAxis, filtered_models, colors,Min, attribut, Name,ax):
    
    # Iterate over each model and its corresponding color
    maxValue = 0
    minValue = 9999999999999
    for i,value in enumerate(attribut):
        value = float(value)
        color = colors[i]
        # Extract attrXAxis and attrYAxis from each model
        xValues = [getattr(model, attrXAxis) for model in filtered_models if getattr(model, Name) == value]
        yValues = [getattr(model, attrYAxis) for model in filtered_models if getattr(model, Name) == value]
        if len(yValues) < 1:
            continue
        maxValue = min(Min,max(maxValue,max(yValues) * 1.1))
        minValue = min(minValue,min(yValues)/1.1)
        # Create the scatter plot for the current model
        ax.scatter(xValues, yValues, color=color) #, label=str(value)
    
    # Set the labels and title
    ax.set_xlabel(attrXAxis)
    ax.set_ylabel(attrYAxis)
    ax.set_title(f"Evolution of {attrYAxis} ") #when varying {attrXAxis}
    ax.set_ylim(minValue, maxValue)
    ax.legend()

def graphs(attrXAxis, filtered_models, colors,attribut, Name):
    # Create subplots for each attrYAxis
    attrYAxisList1 = ["Accuracy", "precision", "recall", "specificity", "falsePositiveRate", "fMeasure"] 
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(12, 12))
    # Flatten the axes array to iterate over each subplot
    axes = axes.flatten()
    # Plot each graph
    for i, attrYAxis in enumerate(attrYAxisList1):
        makeGraph(attrXAxis, attrYAxis, filtered_models, colors,102,attribut, Name, ax=axes[i])

    attrYAxisList2 = ["scoreTruePositive", "scoreTrueNegative", "scoreFalsePositive", "scoreFalseNegative"]
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    # Flatten the axes array to iterate over each subplot
    axes = axes.flatten()
    # Plot each graph
    for i, attrYAxis in enumerate(attrYAxisList2):
        makeGraph(attrXAxis, attrYAxis, filtered_models, colors,999999,attribut, Name, ax=axes[i])

    # Adjust layout and show plots
    plt.tight_layout()
    plt.show()



def main():
    # "llama3_fact" , "mistral_fact", "gemma_fact" ,"lamma3_fact_justification2","gemma_fact_justification2" ,"llama3_fact_justification3","unsloth_model"
    techniques = "basics" # names of the techniques used to improve the llm (not really used)
    techniques = techniques.replace(" ","")
    modelName = ["llama3_fact_justification2bis","llama3_fact_justification2"] # name of the model
    #modelName = ["llama3_fact_fine_tune","llama3_fact_fine_tune2","llama3_fact_fine_tune3"]
    numberOfMovies = [100,300]  # the number of movies the model was tested on
    GPTtemp = 0.2 # temperature of the llm generating the descriptions
    ollamatemp = [1] # temperature of the llm finding the facts
    numberOfGoodFactsCategories = None # number of facts included in the descriptions (not used anymore as now the number of facts is randomised)
    colors = ['blue', 'green','yellow','red','purple'] #colors to be used when differentiating points
    folder_path = "./results/" # path to the folder where results are stored
    
    
    attrXAxis = "Model"
    
    attributForDistinctionInColor = [1]  # will display points in differents colors based on the value of this attribut
    NameOfTheAttributForDistinction = "numberOfAgent" # Name of the attribut above as used in the class file
    
    # Define the criteria as a list of tuples 
    # Comment the criteria that you don't want to use for the filter
    # main calls
    filtered_models = importFiles(folder_path, modelName, numberOfMovies, GPTtemp, numberOfGoodFactsCategories, ollamatemp, techniques )
    if len(filtered_models)!=0:
        graphs(attrXAxis, filtered_models,colors,attributForDistinctionInColor, NameOfTheAttributForDistinction)



if __name__ == "__main__": 
  main()
        






