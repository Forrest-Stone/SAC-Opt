# SAC-Opt: Semantic Anchors for Iterative Correction in Optimization Modeling

> Large language models (LLMs) have opened new paradigms in optimization modeling by enabling the generation of executable solver code from natural language descriptions. Despite this promise, existing approaches typically remain solver-driven: they rely on single-pass forward generation and apply limited post-hoc fixes based on solver error messages, leaving undetected semantic errors that silently produce syntactically correct but logically flawed models. To address this challenge, we propose SAC-Opt, a backward-guided correction framework that grounds optimization modeling in problem semantics rather than solver feedback. At each step, SAC-Opt aligns the original semantic anchors with those reconstructed from the generated code and selectively corrects only the mismatched components, driving convergence toward a semantically faithful model. This anchor-driven correction enables fine-grained refinement of constraint and objective logic, enhancing both fidelity and robustness without requiring additional training or supervision. Empirical results on seven public datasets demonstrate that SAC-Opt improves average modeling accuracy by 7.7\%, with gains of up to 21.9\% on the ComplexLP dataset. These findings highlight the importance of semantic-anchored correction in LLM-based optimization workflows to ensure faithful translation from problem intent to solver-executable code.    

## Dataset

The datasets are download from a recent survey _A Survey of Optimization Modeling Meets LLMs: Progress and Future Directions_. The original datastes download link is [Datasets](https://github.com/LLM4OR/LLM4OR/tree/master/static/clean_benchmarks).


We provide some datasets to get the natural language decription input and true-label values by the file `process_data.py`. And the processed datasets are provided in the `data` folder.


## Run the SAC-Opt

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
