class ModelMetrics:
    def __init__(self, filePath):
        self.techniques = None
        self.numberOfAgent = None
        self.Model = None
        self.GPTtemp = None
        self.ollamatemp = None
        self.Accuracy = None
        self.precision = None
        self.recall = None
        self.specificity = None
        self.falsePositiveRate = None
        self.fMeasure = None
        self.NumberOfMovies = None
        self.NumberOfFacts = None
        self.numberOfFactsCategories = None
        self.numberOfGoodFactsCategories = None
        self.numberOfRightFacts = None
        self.numberOfbadFacts = None
        self.scoreBadOutput = None
        self.scoreTruePositive = None
        self.scoreTrueNegative = None
        self.scoreFalsePositive = None
        self.scoreFalseNegative = None
        # Open the .txt file for reading
        with open(filePath, "r", encoding="utf-8", errors="ignore") as file:
            lines = file.readlines()
        # Iterate through the lines and parse the variables and their values
        for i,line in enumerate(lines):
            parts = line.strip().split(" = ")
            if len(parts) == 2:
                variable_name = parts[0].strip().replace(" ","")
                value = parts[1].strip()
                if variable_name != "Model" and variable_name != "techniques" :
                    value = float(value)  
                if variable_name == "techniques":
                    value = value.replace(" ","")
                setattr(self, variable_name, value)
            if len(parts) == 3:
                variable_name1 = parts[0].strip()
                value = parts[1].strip().split(" and ")
                value1 =float(value[0].strip())
                variable_name2 = value[1].strip()
                value2 =float(parts[2].strip())
                setattr(self, variable_name1, value1)
                setattr(self, variable_name2, value2)
            if i > 23 :
                break
        self.scoreBadOutput = 100 * self.scoreBadOutput / self.NumberOfFacts
    def __str__(self):
        return f"techniques: {self.techniques}, Model: {self.Model}, ollamaTemp: {self.ollamatemp}, NumberOfMovies: {self.NumberOfMovies}, numberOfGoodFactsCategories : {self.numberOfGoodFactsCategories}"
    def model(self):
        return self.Model
