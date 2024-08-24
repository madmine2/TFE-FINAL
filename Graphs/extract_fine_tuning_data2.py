import os
import random
# This code allow you to select some results files depending on some chosen criteria in order to collect data for finetuning
# It will only collect the true positive and true negative answer
folder_path = "./results/"
file_names = os.listdir(folder_path)
fineTuningDataPos = []
fineTuningDataNeg = []
file_path_final = "fineTuningDataImperfect.txt"
# Iterate over each file
for file_name in file_names:
    if file_name.endswith(".txt") :
        # Construct the file path
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            lines = file.readlines()
        if  "llama3Interaction_B" in lines[2]:
            if "numberOfAgent = 1" in lines[1]:
                num = len(lines)
                i =0
                while i < num : 
                    if  "TP  -  TruePositive" in lines[i] or "FP  -  FalsePositive" in lines[i]:
                        temp = lines[i] +"&&&"+ "correct" + "&&&"+ lines[i+2]+"&&&"+  lines[i+4] +'\n'+"@@"  +'\n' 
                        fineTuningDataPos.append(temp)
                        i += 4
                    if  "TN  -  TrueNegative" in lines[i] or "FN  -  FalseNegative" in lines[i]:
                        temp = lines[i] +"&&&"+  "wrong" +"&&&"+  lines[i+2]  + "&&&"+ lines[i+4] +'\n' +"@@" +'\n' 
                        fineTuningDataNeg.append(temp)
                        i += 4
                    i+=1
fineTuningDataPos = fineTuningDataPos[:int(len(fineTuningDataNeg)/8)]  
fineTuningData = fineTuningDataPos + fineTuningDataNeg
random.shuffle(fineTuningData)

with open(file_path_final, "w") as file:
        file.writelines(fineTuningData)
# file_path = 'results/result99 copy.txt'
# with open(file_path, "r") as file:
#             lines = file.readlines()
# specificity_line = "techniques = basic \n"
# recall_line = next(line for line in lines if "Model" in line)
# lines.insert(lines.index(recall_line), specificity_line)
#         # Write the updated lines back to the file
# with open(file_path, "w") as file:
#     file.writelines(lines)