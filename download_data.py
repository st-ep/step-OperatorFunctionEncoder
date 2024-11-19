import os
import urllib.request
import tqdm

pbar, count = None, 0
desc = ""
def show_progress(block_num, block_size, total_size):
    global pbar
    global count
    global desc
    if pbar is None:
        pbar = tqdm.tqdm(total=total_size, unit='B', unit_scale=True, desc=desc)
    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded - count)
        count = downloaded
    else:
        pbar.close()
        pbar = None
        count = 0


os.makedirs("src/Datasets/Heat", exist_ok=True)
os.makedirs("src/Datasets/L-Shaped", exist_ok=True)
os.makedirs("src/Datasets/Darcy", exist_ok=True)
os.makedirs("src/Datasets/Elastic", exist_ok=True)

# download the files
print("Downloading datasets to src/Datasets/...")

##### Heat ##############################################################################################################################################
ln = "https://drive.usercontent.google.com/download?id=1mSMplhocRU_0MJSwAnlXYpo4SCYoSkPk&export=download&authuser=0&confirm=t&uuid=7ef4e753-bfc3-4855-92a2-1ea57e39bf3f&at=APZUnTX9xjPEoRhinxBuhPCVk4iG:1723049906103"
desc = "Heat Train"
urllib.request.urlretrieve(ln, "src/Datasets/Heat/heatequation_train.mat", reporthook=show_progress)

ln = "https://drive.usercontent.google.com/download?id=1JUFD9VDa2Drgd9OZ7yDAGapQ8AJKeaC1&export=download&authuser=0&confirm=t&uuid=9699c394-dae0-45e3-97ba-d5c0462695b8&at=APZUnTXCAZ_BaRILl8N2-q5n2tS_:1723050016071"
desc = "Heat Test"
urllib.request.urlretrieve(ln, "src/Datasets/Heat/heatequation_test.mat", reporthook=show_progress)


##### L-shaped, linear Darcy ##############################################################################################################################
ln = "https://drive.usercontent.google.com/download?id=1dSjOjI-DHhmUgcFonAqLIqvQIfKouqZo&export=download&authuser=0&confirm=t&uuid=90fa75a0-f578-4d6c-856e-99081265b1d3&at=APZUnTVQ7tBgzAs96tmlIzyl9Z9u:1723137673435"
desc = "L-Shaped Train"
urllib.request.urlretrieve(ln, "src/Datasets/L-Shaped/linearDarcy_train.mat", reporthook=show_progress)

ln = "https://drive.usercontent.google.com/download?id=1_6s-fPzTZCqpysLhfocm6qth8OBl1H-k&export=download&authuser=0&confirm=t&uuid=1a727246-2e09-4f82-a65c-fb2234f105b1&at=APZUnTVNqcWgyHb2nmhkk0jydRL9:1723137701171"
desc = "L-Shaped Test"
urllib.request.urlretrieve(ln, "src/Datasets/L-Shaped/linearDarcy_test.mat", reporthook=show_progress)

# #### Elastic ############################################################################################################################################
ln = "https://drive.usercontent.google.com/download?id=1R0-1njHDuMfznhIHKQiXTt6anmTG5BOB&export=download&authuser=0&confirm=t&uuid=02b11c88-689e-4034-8846-6fcfd7d952eb&at=AENtkXb_iwzibWhNxiG8Ni4unt9b:1731994146279"
desc = "Elastic Train"
urllib.request.urlretrieve(ln, "src/Datasets/Elastic/linearElasticity_train.mat", reporthook=show_progress)

ln = "https://drive.usercontent.google.com/download?id=1R0-1njHDuMfznhIHKQiXTt6anmTG5BOB&export=download&authuser=0&confirm=t&uuid=02b11c88-689e-4034-8846-6fcfd7d952eb&at=AENtkXb_iwzibWhNxiG8Ni4unt9b:1731994146279"
desc = "Elastic Test"
urllib.request.urlretrieve(ln, "src/Datasets/Elastic/linearElasticity_test.mat", reporthook=show_progress)


print("Done!\n\n")