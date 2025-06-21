import csv

# Read the CSV file
with open('tennis.csv', 'r') as f:
    reader = csv.reader(f)
    your_list = list(reader)

# Initialize the most specific hypothesis
h = ['0', '0', '0', '0', '0', '0']

# Iterate through each training example
for i in your_list:
    print(i)
    if i[-1] == "True":  # Check if the example is positive
        for j in range(len(i) - 1):  # Exclude the label
            if h[j] == '0':
                h[j] = i[j]
            elif h[j] != i[j]:
                h[j] = '?'

print("\nMost specific hypothesis is:")
print(h)
