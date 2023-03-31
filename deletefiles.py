import os
from typing import List 

def delete_files(file_dir: str, start:int, end:int):
    '''
    Delete files in given range, from given folder
    '''
    for i in range(start, end+1):
        file_name = str(i) + suffix
        file_path = f"{file_dir}/{file_name}"
        if os.path.exists(file_path):
            os.remove(file_path)
        else:
            print("The file does not exist")

file_dir = "/Users/huayinluo/Desktop/code/CaTracking/original/11415" # change to your directory

# files to delete
files_delete = [
    (0,3),
    (21,23),
    (342, 352)
]

# suffix of image files
suffix = ".png"
    
# delete files in given ranges
for start, end in files_delete:
    print(f"start delete range {start} - {end}")
    delete_files(file_dir, start, end)
    print(f"finish delete range {start} - {end}")
