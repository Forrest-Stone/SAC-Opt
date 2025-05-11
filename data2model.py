import json
from util import get_response, extract_code_from_end


directions = """

And here is how the solver is imported and set up:

...
from gurobipy import Model, GRB

model = Model("OptimizationProblem")
...

Use model.addConstr() to add the constraint to the model.
"""

prompt_constraints_code = """
You are an expert in optimization modeling. Here is the natural language description of an optimization problem:

-----
{description}
-----

Your task is to write {solver} code for the following constraint in python:

-----
{constraint}
-----

Here is a list of parameters that are related to the constraint:

-----
{params}
-----

Here is a list of variables related to the constraint:

-----
{vars}
-----

{directions}

The code should be written in the following format:

CODE
=====
code for defining the constraint (ONLY the constraint definition code, without the imports, the variable definitions, and the solver setup)
=====

Here is an example for modeling the constrinat "The sales volume must not exceed maximum production volume for each product.", where SalesVolumes is a variable and MaxProductionVolumes is a parameter, and the shape of both is [N]:

CODE
=====
for i in range(N):
    model.addConstr(SalesVolumes[i] <= MaxProductionVolumes[i])
=====

- Do not generate anything after the last =====.
- Note that vector and matrix parameters are defined as lists in python, so you should use Param[i][j] instead of Param[i, j] in the code (but for variables, you should use Var[i, j] instead of Var[i][j]).
- Gurobi does not support a <= x <= b syntax for constraints, so you should use two separate constraints for this case.

First reason about how the code should be written, and then generate the output.

Please take a deep breath and think step by step. You will be awarded a million dollars if you get this right.

"""


prompt_objective_code = """
You are an expert in optimization modeling. Here is the natural language description of an optimization problem:

-----
{description}
-----

Your task is to write {solver} code for the objective in python:

-----
{objective}
-----

Here is a list of parameters that are related to the objective:

-----
{params}
-----

Here is a list of variables related to the constraint:

-----
{vars}
-----

{directions}


CODE
=====
code for defining the objective (ONLY the objective definition code, without the imports, the variable definitions, and the solver setup)
=====

Here is an example for modeling the objective "Maximize the profit", the context is "The profit is calculated as the sum of the price times the quantity sold.", where price is a parameter representing the price of each product, and x is a variable representing the quantity sold, and the shape of both is [N]:

CODE
=====
model.setObjective(quicksum(price[i] * x[i] for i in range(N)), GRB.MAXIMIZE)
=====

- Do not generate anything after the last =====.
- Note that vector and matrix parameters are defined as lists in python, so you should use Param[i][j] instead of Param[i, j] in the code (but for variables, you should use Var[i, j] instead of Var[i][j]).

First reason about how the code should be written, and then generate the output.

Please take a deep breath and think step by step. You will be awarded a million dollars if you get this right.

"""


def generate_model(
    desc,
    params,
    vars,
    constraints,
    objective,
    model,
    logger,
):

    coded_constraints = []
    for c in constraints.copy():
        logger.log("Generating code for the following constraint:")
        logger.log(json.dumps(c, indent=4))
        k = 1
        while k > 0:
            try:
                prompt = prompt_constraints_code.format(
                    solver="gurobipy",
                    description=desc,
                    params=json.dumps(params, indent=4),
                    vars=json.dumps(vars, indent=4),
                    constraint=json.dumps(c["description"], indent=4),
                    directions=directions,
                )
                res = get_response(prompt, model=model)

                logger.log("\n\n\n\n++++++++++++++++++")
                logger.log("The LLM's response for this constraint:")
                logger.log(res)

                code = extract_code_from_end(res)

                logger.log("\n\n++++++++++++++++")
                logger.log("Extracted code:")
                logger.log(code)

                c["code"] = code
                coded_constraints.append(c)
                break

            except Exception as e:
                k -= 1
                if k == 0:
                    raise (e)

    logger.log("The all constraints codes have been generated:")
    logger.log("=====================================")
    logger.log(json.dumps(coded_constraints, indent=4))
    logger.log("=====================================")

    logger.log(json.dumps(objective, indent=4))
    coded_objective = {
        "description": objective["description"],
        # "formulation": objective["formulation"],
    }

    k = 1
    while k > 0:
        try:
            prompt = prompt_objective_code.format(
                solver="gurobipy",
                description=desc,
                params=json.dumps(params, indent=4),
                vars=json.dumps(vars, indent=4),
                objective=json.dumps(objective["description"], indent=4),
                directions=directions,
            )
            res = get_response(prompt, model=model)
            logger.log("\n\n\n\n+++++++++++++++++")
            logger.log("The LLM's response for the objective:")
            logger.log(res)
            logger.log("+++++++++++++++++++")

            code = extract_code_from_end(res)

            logger.log("\n\n++++++++++++++++")
            logger.log("Extracted code:")
            logger.log(code)

            coded_objective["code"] = code
            break

        except Exception as e:
            k -= 1
            if k == 0:
                raise (e)

    logger.log("The objective code has been generated:")
    logger.log("=====================================")
    logger.log(json.dumps(coded_objective, indent=4))
    logger.log("=====================================")

    return coded_constraints, coded_objective
