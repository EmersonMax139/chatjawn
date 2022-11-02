import os

with open(os.path.join('../data/comments', 'comments_long.txt'), 'r') as text_file:
    lines = text_file.readlines() 
with open(os.path.join('../data/comments', 'comments_long_cleaned.txt'), 'w') as text_file:
    for line in lines: 
        if len(line.split(' ')) < 15:
            text_file.write(line)
