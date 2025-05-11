import json
import openai
import numpy as np
import os


def load_state(state_file):
    with open(state_file, "r") as f:
        state = json.load(f)
    return state


def save_state(state, dir):
    with open(dir, "w") as f:
        json.dump(state, f, indent=4)


def create_state(parent_dir, run_dir):
    # read params.json
    with open(os.path.join(parent_dir, "params.json"), "r") as f:
        params = json.load(f)

    data = {}
    for key in params:
        data[key] = params[key]["value"]
        del params[key]["value"]

    # save the data file in the run_dir
    with open(os.path.join(run_dir, "data.json"), "w") as f:
        json.dump(data, f, indent=4)

    # read the description
    with open(os.path.join(parent_dir, "description.txt"), "r") as f:
        desc = f.read()

    state = {"description": desc, "parameters": params}
    return state


class Logger:
    def __init__(self, file):
        self.file = file

    def log(self, text):
        with open(self.file, "a") as f:
            f.write(text + "\n")

    def reset(self):
        with open(self.file, "w") as f:
            f.write("")


def prep_problem_json(state):
    for parameter in state["parameters"]:
        assert "shape" in parameter.keys(), "shape is not defined for parameter"
        assert "symbol" in parameter.keys(), "symbol is not defined for parameter"
        assert (
            "definition" in parameter.keys() and len(
                parameter["definition"]) > 0
        ), "definition is not defined for parameter"

        if parameter["shape"]:
            code_symbol = parameter["symbol"].split("_")[0]
            parameter["code"] = (
                f'{code_symbol} = np.array(data["{code_symbol}"]) # {parameter["shape"]}'
            )
        else:
            code_symbol = parameter["symbol"].split("_")[0]
            parameter["code"] = (
                f'{code_symbol} = data["{code_symbol}"] # scalar parameter'
            )

    return state


def sanity_check(state):
    # read the data from file:

    assert (
        "data_json_path" in state.keys()
    ), "data_json_path is not defined in the state"
    print("data_json_path is not defined in the state")

    with open(state["data_json_path"], "r") as f:
        data = json.load(f)

    for param in state["parameters"]:
        if not "shape" in param.keys():
            print(f"shape is not defined for parameter {param['definition']}")
            raise KeyError(
                f"shape is not defined for parameter {param['definition']}")

        if not "symbol" in param.keys():
            print(f"symbol is not defined for parameter {param['definition']}")
            raise KeyError(
                f"symbol is not defined for parameter {param['definition']}")

        if len(param["symbol"].split("_")) > 2:
            print(
                f"Please use camelCase for parameter symbols! Error in {param['symbol']}")
            raise KeyError(
                f"Please use camelCase for parameter symbols! Error in {param['symbol']}"
            )

        if not "definition" in param.keys():
            print(
                f"definition is not defined for parameter {param['definition']}")
            raise KeyError(
                f"definition is not defined for parameter {param['definition']}"
            )

        symb = param["symbol"].split("_")[0]
        if not symb in data.keys():
            print(f"{param['symbol']} is not defined in data.json")
            raise KeyError(f"{param['symbol']} is not defined in data.json")

        pd = np.array(data[symb])
        if param["shape"] != []:
            for idx, dim in enumerate(param["shape"]):
                if not dim in data:
                    print(f"{dim} is not defined in data.json")
                    raise KeyError(
                        f"{dim} is not defined in data.json, but is used in {param['symbol']}"
                    )

                if data[dim] != pd.shape[idx]:
                    print(
                        f"Dimension mismatch for {param['symbol']} at dim {idx}"
                    )
                    raise ValueError(
                        f"Dimension mismatch for {param['symbol']} at dim {idx}"
                    )

    # make sure that parameter codes are defined
    for param in state["parameters"]:
        if not "code" in param.keys():
            print(f"code is not defined for parameter {param['definition']}")
            raise KeyError(
                f"code is not defined for parameter {param['definition']}")


def get_openai_client():
    with open("config.json") as f:
        config = json.load(f)
    if len(config["openai_api_key"]) < 10:
        raise ValueError(
            "Please provide a valid OpenAI API key in config.json")
    config["openai_api_key"]

    client = openai.Client(
        api_key=config["openai_api_key"],
        base_url=config["base_url"]
    )

    return client


def extract_code_from_end(text):
    # get 1st and 2nd occurence of "====="
    if "=====" in text:
        ind_1 = text.find("=====")
        ind_2 = text.find("=====", ind_1 + 1)

        code = text[ind_1 + len("====="): ind_2].strip()
    else:
        ind_1 = text.find("```python")
        ind_2 = text.find("```", ind_1 + 1)

        code = text[ind_1 + len("```"): ind_2].strip()

    if "```" in code:
        code = code.replace("```python", "").replace("```", "").strip()

    if code.startswith("====="):
        code = code[len("====="):].strip()

    if code.endswith("====="):
        code = code[: -len("=====")].strip()

    if "python" in code:
        code = code.replace("python", "").strip()

    return code


def extract_list_from_end(text):
    ind = len(text) - 1
    while text[ind] != "]":
        ind -= 1
    text = text[: ind + 1]

    ind -= 1
    cnt = 1
    while cnt > 0:
        if text[ind] == "]":
            cnt += 1
        elif text[ind] == "[":
            cnt -= 1
        ind -= 1

    # convert to json format
    jj = json.loads(text[ind + 1:])
    return jj


def extract_json_from_end(text):

    try:
        return extract_json_from_end_backup(text)
    except:
        pass

    # Find the start of the JSON object
    json_start = text.find("{")
    if json_start == -1:
        raise ValueError("No JSON object found in the text.")

    # Extract text starting from the first '{'
    json_text = text[json_start:]

    # Remove backslashes used for escaping in LaTeX or other formats
    json_text = json_text.replace("\\", "")

    # Remove any extraneous text after the JSON end
    ind = len(json_text) - 1
    while json_text[ind] != "}":
        ind -= 1
    json_text = json_text[: ind + 1]

    # Find the opening curly brace that matches the closing brace
    ind -= 1
    cnt = 1
    while cnt > 0 and ind >= 0:
        if json_text[ind] == "}":
            cnt += 1
        elif json_text[ind] == "{":
            cnt -= 1
        ind -= 1

    # Extract the JSON portion and load it
    json_text = json_text[ind + 1:]

    # Attempt to load JSON
    try:
        jj = json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to decode JSON: {e}")

    return jj


def extract_json_from_end_backup(text):

    if "```json" in text:
        text = text.split("```json")[1]
        text = text.split("```")[0]
    ind = len(text) - 1
    while text[ind] != "}":
        ind -= 1
    text = text[: ind + 1]

    ind -= 1
    cnt = 1
    while cnt > 0:
        if text[ind] == "}":
            cnt += 1
        elif text[ind] == "{":
            cnt -= 1
        ind -= 1

    # find comments in the json string (texts between "//" and "\n") and remove them
    while True:
        ind_comment = text.find("//")
        if ind_comment == -1:
            break
        ind_end = text.find("\n", ind_comment)
        text = text[:ind_comment] + text[ind_end + 1:]

    # convert to json format
    jj = json.loads(text[ind + 1:])
    return jj


def extract_equal_sign_closed(text):
    ind_1 = text.find("=====")
    ind_2 = text.find("=====", ind_1 + 1)
    obj = text[ind_1 + 6: ind_2].strip()
    return obj


def shape_string_to_list(shape_string):
    if type(shape_string) == list:
        return shape_string
    # convert a string like "[N, M, K, 19]" to a list like ['N', 'M', 'K', 19]
    shape_string = shape_string.strip()
    shape_string = shape_string[1:-1]
    shape_list = shape_string.split(",")
    shape_list = [x.strip() for x in shape_list]
    shape_list = [int(x) if x.isdigit() else x for x in shape_list]
    if len(shape_list) == 1 and shape_list[0] == "":
        shape_list = []
    return shape_list


def get_response(prompt, model="gpt-4o"):
    client = get_openai_client()
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
            {
                "role": "system",
                "content": "You're a helpful assistant.",
            }
        ],
        model=model,
        temperature=0,
        seed=42,
    )

    res = chat_completion.choices[0].message.content
    return res
