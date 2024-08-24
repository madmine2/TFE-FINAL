import os
# This code allow you to select some results files depending on some chosen criteria in order to collect data for finetuning
# It will only collect the true positive and true negative answer
folder_path = "../results/"
file_names = os.listdir(folder_path)
fineTuningData = []
file_path_final = "fineTuningData2.txt"
# Iterate over each file
for file_name in file_names:
    if file_name.endswith(".txt") :
        # Construct the file path
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "r") as file:
            lines = file.readlines()
        if  "gemma_fact" in lines[2] or "llama3_fact" in lines[2] and not "justification" in  lines[2]:
            if "numberOfAgent = 1" in lines[1]:
                num = len(lines)
                i =0
                while i < num : 
                    if  "TP  -  TruePositive" in lines[i] :
                        temp = lines[i] + lines[i+1]  + lines[i+2]+ lines[i+3]  + lines[i+4] +'\n'+"@"  +'\n' 
                        fineTuningData.append(temp)
                        i += 4
                    if  "TN  -  TrueNegative" in lines[i]:
                        temp = lines[i] + lines[i+1] + lines[i+2]  + lines[i+3] + lines[i+4] +'\n' +"@" +'\n' 
                        fineTuningData.append(temp)
                        i += 4
                    i+=1

        
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