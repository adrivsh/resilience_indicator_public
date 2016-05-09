import sys
from IPython.display import clear_output
def progress_reporter(c):
    #report progress
    clear_output()
    print("working on ",c)
    sys.stdout.flush()    