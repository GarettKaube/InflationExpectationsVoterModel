# InflationExpectationsVoterModel

Before running run:
"pip install -r requirements.txt" in a terminal

Run the file with prompt such as: "python vm.py --population 100 --edges 200 --data r --num_runs 50"

vm.py arguments:

1. --population: Number of nodes/individuals in model (int), default = 10
2. --edges: Number of edges for random grap, population <= edges (int), default = 10
3. --data: r or p where r=random initial expectations, p=predifined data in text file, default = r
4. --num_runs: number of runs for the voter model, default = 10
5. --torch: use torch model, default = False/n


Google Colab notebook: https://colab.research.google.com/drive/1I9kJMLVg3GqH2WddDxHzlaRe_hUua8ri?usp=sharing
