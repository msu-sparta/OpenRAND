import subprocess
from datetime import datetime
import os

# Get current date and time
now = datetime.now()

# Format and print it
print("Current date and time:", now.strftime("%Y-%m-%d %H:%M:%S"))

ROOT_DIR = os.getcwd()
BUILD_DIR = os.path.join(ROOT_DIR, "build")
RES_DIR = os.path.join(BUILD_DIR, "results")
print(ROOT_DIR, BUILD_DIR, RES_DIR)

procs = []
# create results directory if not exists
os.makedirs(RES_DIR, exist_ok=True)


# launch practrand multistream tests
PRACT_RND_EXEC = os.path.join(BUILD_DIR, "Practrand", "RNG_test")

for gen in ["philox", "tyche", "threefry", "squares"]:
    command  = f"{BUILD_DIR}/tests/pract_rand_multi {gen} | {PRACT_RND_EXEC} stdin32 -multithreaded -tlmax 8GB > {RES_DIR}/practrandm_{gen}.txt"

    p = subprocess.Popen(command, shell=True)
    p.name = f"practrand_{gen}"
    procs.append(p)



# launch TESTU01 tests
# LEVEL = "crush"

# for gen in ["philox", "tyche", "threefry", "squares"]:
#     command  = f"{BUILD_DIR}/tests/testu01_multi {gen} {LEVEL} > {RES_DIR}/testu01m_{gen}.txt"
#     print(command)
#     p = subprocess.Popen(command, shell=True)
#     procs.append(p)

# Wait for them to finish
for p in procs:
    p.communicate()
    if p.returncode != 0:
        print("Error in test ", p.name)
        exit(1)
    print(p.name, " finished successfully")   
    now = datetime.now()
    print("Current date and time:", now.strftime("%Y-%m-%d %H:%M:%S"))
