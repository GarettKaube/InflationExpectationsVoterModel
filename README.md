# Inflation Expectations Voter Model

Before running run:
"pip install -r requirements.txt" in a terminal

Run the file with prompt such as: "python vm.py --population 100 --edges 200 --data r --num_runs 50"

vm.py arguments:

1. --population: Number of nodes/individuals in model (int), default = 10
2. --edges: Number of edges for random grap, population <= edges (int), default = 10
3. --data: rn, rexp or p where rn= normally distributed expectations, rexp= exponetially distributed expectations, p=predifined data in text file, default = rn
4. --num_runs: number of runs for the voter model, default = 10
5. --torch: use pytorch model, default = False/n
6. --iterations: Number of iterations per run of the voter model, default = 500

Google Colab notebook: https://colab.research.google.com/drive/1_A0GpLUeM5SXitBffwY6B324Xl-U9vMK?usp=sharing
