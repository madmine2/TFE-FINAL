import os
# This code take all .txt files in a folder and allow you to make a modification to add a metrics retro actively,
# #here it's for specificity but it can be adapted
folder_path = "./results/"
file_names = os.listdir(folder_path)
# Iterate over each file
for file_name in file_names:
    if file_name.endswith(".txt"):
        # Construct the file path
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "r") as file:
            lines = file.readlines()


        specificity_line = "numberOfAgent = 1 \n"
        recall_line = next(line for line in lines if "Model" in line)
        lines.insert(lines.index(recall_line), specificity_line)

        # # Find recall line and extract its value
        # recall_line = next(line for line in lines if "recall" in line)
        # recall_value = float(recall_line.split("=")[1].strip())


        # # Find the index of the lines containing true negatives and false positives
        # true_negatives_index = next(i for i, line in enumerate(lines) if "scoreTrueNegative" in line)
        # false_positives_index = next(i for i, line in enumerate(lines) if "scoreFalsePositive" in line)

        # # Extract true negatives and false positives values
        # true_negatives = int(lines[true_negatives_index].split()[-1])
        # false_positives = int(lines[false_positives_index].split()[-1])
        # # Calculate specificity
        # specificity = true_negatives / (true_negatives + false_positives) *100

        # # Update the file with specificity
        # specificity_line = f"specificity = {specificity}\n"
        # lines.insert(lines.index(recall_line) + 1, specificity_line)
        
        with open(file_path, "w") as file:
            file.writelines(lines)
# file_path = 'results/result99 copy.txt'
# with open(file_path, "r") as file:
#             lines = file.readlines()
# specificity_line = "techniques = basic \n"
# recall_line = next(line for line in lines if "Model" in line)
# lines.insert(lines.index(recall_line), specificity_line)
#         # Write the updated lines back to the file
# with open(file_path, "w") as file:
#     file.writelines(lines)