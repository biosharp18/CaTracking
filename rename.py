import os
import natsort
from typing import List

def rename_imgs(sorted_filenames: List, file_dir: str):
    '''
    rename files from "ORIGINALFRAME.png" -> "NEWNUM_ORIGINALFRAME.png"
    '''
    for filename in sorted_filenames:
        frame = filename.split(".")[0].split("_")[1]
        new_name = f"{frame}.png"
        os.rename(f"{file_dir}/{filename}", f"{file_dir}/{new_name}")
        
def revert_rename_imgs(sorted_filenames, file_dir):
    '''
    rename files to original names ("NEWNUM_ORIGINALFRAME.png" -> "ORIGINALFRAME.png")
    use if made mistake in naming (ie. remove/add images)
    '''
    i=0
    for filename in sorted_filenames:
        frame = filename.split(".")[0]
        new_name = f"{i}_{frame}.png"
        os.rename(f"{file_dir}/{filename}", f"{file_dir}/{new_name}")
        i+=1
    

file_dir = "/Users/huayinluo/Desktop/code/CaTracking/masked/11415" # change to your directory
sorted_filenames = natsort.natsorted(os.listdir(file_dir),reverse=False) # sort files in order

mistake=False
    
if mistake:
    rename_imgs(sorted_filenames, file_dir)
        
else:
    revert_rename_imgs(sorted_filenames, file_dir)