# SAC-Opt: Semantic Anchors for Iterative Correction in Optimization Modeling

>



## Dataset

The datasets are download from a recent survey _A Survey of Optimization Modeling Meets LLMs: Progress and Future Directions_. The original datastes download link is [Datasets](https://github.com/LLM4OR/LLM4OR/tree/master/static/clean_benchmarks).


We provide some datasets to get the natural language decription input and true-label values by the file `process_data.py`. And the process datasets are provided in the `data` folder.


## Run the SCOpt

### Setup the Packages and OpenAI API

1. Install and setup Python Packages.
2. Install and setup "Gurobi Optimizer" from [Gurobi's official website](https://www.gurobi.com/downloads/gurobi-software/).
3. Setup your own api_key in the `config.json` file.

### Make sure the generated data.json path correct

Update line **171** of the `run.py` file or **190** of the `run_abstract.py` file to ensure that the absolute address path of the generated data.json file is correct.

### Run the Code

For the concrete models (nl4opt, industryor, easylp, complexlp, nlp4lp, resocratic), where the problem description includes the parameters, we execute the code using:

`
python run.py --dataset nlp4lp --problem 0
`


For the abstract model (complexor), where the parameters are separate from the problem description, we run the code as follows:


`python run_abstract.py --dataset complexor --problem aircraft_assignment
`
