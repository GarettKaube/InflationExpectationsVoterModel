Before running run:
"pip install -r requirements.txt" in a terminal

Run the file with prompt such as: "python vm.py --population 100 --edges 200 --data r --num_runs 50"

vm arguments:

--population: Number of nodes/individuals in model (int), default = 10
--edges: Number of edges for random grap, population <= edges (int), default = 10
--data: r or p where r=random initial expectations, p=predifined data in text file, default = r
--num_runs: number of runs for the voter model, default = 10
--torch: use torch model, default = False