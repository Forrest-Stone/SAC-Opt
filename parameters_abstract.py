import os
import json
import re

from util import extract_json_from_end, get_response, shape_string_to_list


def extract_score(text, params, param):
    match = re.search(r"\d out of 5", text.lower())
    if match:
        score = int(match.group()[0])
        if score > 3:
            return True, params
        else:
            ## delete the parameter of score <= 3
            del params[param]
            return False, params
            # inp = input(
            #     "LLMs reasoning: {}\n Do you want to keep parameter {}? (y/n): ".format(
            #         text, param
            #     ),
            # )
            # if inp == "y":
            #     return True, params
            # else:
            #     del params[param]
            #     return True, params
    else:
        return False, None


prompt_params = """
You are an expert in optimization modeling. Here is the natural language description of an optimization problem:

-----
{description}
-----

Here is the input data for the problem:

-----
{input_data}
-----

Your task is to identify and extract parameters from the description. The parameters are only explictly mentioned values that are already known. The {input_data} is a list of parameters that someone has extracted from the description. You can use them to help you identify the parameters. Please generate the output in the following format:

{{
    "SYMBOL": {{
        "definition": "DEFINITION",
        "shape": "SHAPE",
        "value": "VALUE"
    }}
}}

Where SYMBOL is a string representing the parameter (use CamelCase), SHAPE is the shape of the parameter (e.g. "[]" for scalar, or ["N", "M"] for a matrix of size N x M where N and M are scalar parameters), DEFINITION is a string describing the parameter, and VALUE is the explictly mentioned value of the parameter, and the type of value is only one of "int", "float", or "binary", NOT "string". For example,

{{
    "NumberOfItems": {{
        "shape": "[]",
        "definition": "The number of items in the inventory",
        "value": 5
    }},
    "ItemValue": {{
        "shape": ["NumberOfItems"],
        "definition": "The value of each item in the inventory",
        "value": [10.0, 20.0, 30.0, 40.0, 50.0]
    }}
}}

- Put all the parameters in a single json object.
- Do not generate anything after the json object.
- Ensure referenced parameters (e.g., N used in shape definitions) are defined before their dependents.
- Ensure array lengths match dimension parameter values.
- Ensure that the parameters are only the ones that are explicitly mentioned in the description.
- Do not include any other information or reasoning in your response.

Take a deep breath and think step by step. You will be awarded a million dollars if you get this right.

"""

prompt_params_q = """
You are an expert in optimization modeling. Here is the natural language description of an optimization problem:

-----
{description}
-----

Here is a list of parameters that someone has extracted from the description:

-----
{params}
-----

Consider parameter "{targetParam}".

{question}

"""

qs = [
    (
        """
- Is the value of it already known or not? based on that, how confident are you that this is a parameter (from 1 to 5)?
- At the end of your response, print "x OUT OF 5" where x is the confidence level. Do not generate anything after that.

Please take a deep breath and think step by step. You will be awarded a million dollars if you get this right.

""",
        extract_score,
    ),
]


def get_params(desc, orign_data, check=True):

    k = 5
    while k > 0:
        try:
            res = get_response(prompt_params.format(description=desc, input_data=orign_data))
            params = extract_json_from_end(res)
            break
        except:
            k -= 1
            if k == 0:
                raise Exception("Failed to extract parameters")

    if check:
        for q, func in qs:
            for param in params.copy():
                k = 5
                while k > 0:
                    prompt = prompt_params_q.format(
                        description=desc,
                        params=json.dumps(params, indent=4),
                        question=q,
                        targetParam=param,
                    )
                    x = get_response(prompt)

                    print(x)
                    print("-------")

                    valid, res = func(x, params, param)
                    if valid:
                        params = res
                        break
                    else:
                        k -= 1

    for p in params:
        params[p]["shape"] = shape_string_to_list(params[p]["shape"])
    return params
