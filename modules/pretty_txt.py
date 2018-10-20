import sys
import os

if len(sys.argv) is 1:
    print("no file name")
    exit(0)
    
file_name = sys.argv[1]
text = ""
#file_name = "01.06.Feature_extration_from_text_and_images.01.Bag_of_words.txt"
backup_file_name = file_name.replace('.txt', '.backup.txt')
with open (file_name, "r+") as read_file:
    text=''.join(read_file.readlines()).replace('[MUSIC]', '').replace('\n', ' ').replace('. ', '.\n').strip()

os.rename(file_name, backup_file_name)

with open(file_name, 'w') as write_file:
    write_file.write(text)