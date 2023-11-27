import json 
import argparse 
import random 

parser = argparse.ArgumentParser()
parser.add_argument("--result_file", type=str)

args = parser.parse_args()

with open(args.result_file, 'r') as f:
    data = json.load(f)
"The answer is "

acc = 0
for each in data:
    if each['answer_letter'] == each['response'].strip().replace(".", ""):
        acc += 1
    if each['response'][14] in "ABCDE":
        acc += each['response'][14] == each['answer_letter']
    else:
        acc += random.choice("ABCDE") == each['answer_letter']

print("Acc: {:.2f}%".format(acc / len(data) * 100)) 