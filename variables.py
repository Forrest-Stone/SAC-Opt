import json
import re
from util import extract_json_from_end, get_response


def extract_score(text, vars, var):
    match = re.search(r"\d out of 5", text.lower())
    if match:
        score = int(match.group()[0])
        if score > 3:
            return True, vars
        else:
            ## delete the variable of score <= 3
            del vars[var]
            return False, vars
            # inp = input(
            #     "LLMs reasoning: {}\n Do you want to keep variable {}? (y/n): ".format(
            #         text, var
            #     ),
            # )
            # if inp == "y":
            #     return True, vars
            # else:
            #     del vars[var]
            #     return True, vars
    else:
        return False, None


prompt_vars = """
You are an expert in optimization modeling. Here is the natural language description of an optimization problem:

-----
{description}
-----

And here is a list of parameters that we have extracted from the description:

-----
{params}
-----

Your task is to identify and extract variables from the description. The variables are values that are not known and need to be determined by the optimization model. Please generate the output in the following format:

{{
    "SYMBOL": {{
        "shape": "SHAPE",
        "type": "TYPE",
        "definition": "DEFINITION"
    }}
}}

Where SYMBOL is a string representing the variable (use CamelCase), SHAPE is the shape of the variable (e.g. [] for scalar, or ["N", "M"] for a matrix of size N x M where N and M are scalar parameters), and TYPE is the type of the variable (e.g. "continuous" or "binary", or "integer") and DEFINITION is a string describing the variable. For example:

{{
    "MoneySpent": {{
        "shape": ["N"],
        "type": "continuous",
        "definition": "The amount of money spent on each item in the inventory"
    }},
}}

- Put all the parameters in a single json object.
- Do not generate anything after the json object.

Please take a deep breath and think step by step. You will be awarded a million dollars if you get this right.

"""

prompt_vars_q = """
You are an expert in optimization modeling. Here is the natural language description of an optimization problem:

-----
{description}
-----

Here is a list of parameters that we has extracted from the description:

-----
{params}
-----

Here is a list of variables that we has extracted from the description:

-----
{vars}
-----

Consider variable "{targetVar}".

{question}

"""

qs = [
    (
        """
- Is the variable of it already known or not? based on that, how confident are you that this is a variable (from 1 to 5)?
- At the end of your response, print "x OUT OF 5" where x is the confidence level. Do not generate anything after that.

Please take a deep breath and think step by step. You will be awarded a million dollars if you get this right.

""",
        extract_score,
    ),
]


def get_vars(desc, params, vars=None, check=False):
    if not vars:
        res = get_response(
            prompt_vars.format(
                description=desc, params=json.dumps(params, indent=4)),
        )

        vars = extract_json_from_end(res)

    if check:
        for q, func in qs:
            for var in vars.copy():
                k = 5
                while k > 0:
                    prompt = prompt_vars_q.format(
                        description=desc,
                        params=json.dumps(params, indent=4),
                        vars=json.dumps(vars, indent=4),
                        question=q,
                        targetVar=var,
                    )
                    # print(prompt)
                    x = get_response(prompt)

                    # print(json.dumps(vars[var], indent=4))
                    print(x)
                    print("-------")

                    valid, res = func(x, vars, var)
                    if valid:
                        vars = res
                        break
                    else:
                        k -= 1

                # print("------")

    return vars
