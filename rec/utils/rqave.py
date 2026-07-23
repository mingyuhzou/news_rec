import os
import datetime
def delete_file(filename):
    if os.path.exists(filename):
        os.remove(filename)
def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)

def get_local_time():
    cur=datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")

    return cur