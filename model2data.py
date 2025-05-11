import json
from util import get_response, extract_code_from_end

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


# Initialize the semantic model
st_model = SentenceTransformer("all-MiniLM-L6-v2")


def is_semantically_consistent(text1, text2, threshold=0.75):
    """Use cosine similarity to determine semantic consistency"""
    ans_check = False
    embeddings = st_model.encode([text1, text2])
    similarity = cos_sim(embeddings[0], embeddings[1]).item()
    if similarity >= threshold:
        ans_check = True
    return ans_check


prompt_constraints_language = """
You are an expert in optimization modeling. Here is the natural language description of an optimization problem:

{description}

You are given a constraint implemented in {solver} code and an example natural language description that serves only as a reference for sentence structure and length. Your task is to generate a **new** natural language description that:


1. **Is derived strictly from the given code** — do not assume information not present in the code.
2. **Maintains the structure, length, and complexity of the example description**, but is reworded.
3. **Does not directly copy the example text** — use a natural rephrasing while preserving accuracy.

The example description for the constraint is (For Structure & Length Reference Only, NOT for Content Copying):

-----
{constraint}
-----

Here is the code for the constraint:

-----
{constraint_code}
-----

Here is a list of parameters that are related to the constraint:

-----
{params}
-----

Here is a list of variables related to the constraint:

-----
{vars}
-----

The new description should be written in the following format:

CONSTRAINT:
=====
new natural language description for translating the constraint. (The description should be fully based on the code and should match the structure and length of the example description.)
=====

- Do not generate anything after the last =====.
- Do not include any additional information or explanations.

First reason about how the natural language description should be written, and then generate the output.

Please take a deep breath and think step by step. You will be awarded a million dollars if you get this right.

"""


prompt_constraints_language_coverage = """
You are an expert in optimization modeling.

You task is to judge the consistency of the new generated description and the original description of the same constraint.

The original description is:
-----
{constraint}
-----

The new description is:
-----
{constraint_new}
-----

Please respond with "YES" if the two descriptions are consistent, and "NO" if they are not.

The asnwer should be in the following format:

ANSWER:
=====
YES or NO (ONLY one word and the answer should be in capital letters)
=====

- Do not generate anything after the last =====.
- Do not include any additional information or explanations.

Please take a deep breath and think step by step. You will be awarded a million dollars if you get this right.

"""


def generate_data(
    desc,
    params,
    vars,
    constraints,
    model,
    logger,
    coverage="prompt",
):

    languaged_constraints = []
    # check if we need to update the constraints
    update_constraints = False
    # check which number of constraints we need to update
    error_count = 0
    for c in constraints.copy():
        logger.log(
            "We will generate the description for the following constraint:")
        logger.log(json.dumps(c, indent=4))

        if c["error"] == "NO":
            languaged_constraints.append(c)
            logger.log(
                "For this constraint, the generated description is correct, so we do not need to generate the new description.")
            continue
        error_count += 1
        k = 1
        while k > 0:
            try:
                prompt = prompt_constraints_language.format(
                    description=desc,
                    solver="gurobipy",
                    constraint=json.dumps(constraints, indent=4),
                    constraint_code=json.dumps(c["code"], indent=4),
                    params=json.dumps(params, indent=4),
                    vars=json.dumps(vars, indent=4),
                    # constraint=json.dumps(c["code"], indent=4),
                )
                res = get_response(prompt, model=model)

                logger.log("For " + str(error_count) +
                           "th constraint to generate the decription")

                logger.log("\n\n\n\n++++++++++++++")
                logger.log("The LLM's response for this constraint:")
                logger.log(res)
                logger.log("++++++++++++++")

                code = extract_code_from_end(res)

                logger.log("\n\n++++++++++++")
                logger.log("Extracted new description for the constraint:")
                logger.log(code)
                logger.log("++++++++++++")

                c["description_new"] = code
                languaged_constraints.append(c)

                # check the constraint is correct: have two methods, prompt and math to calculate the constraint old and new
                if coverage == "prompt":
                    coverage_check = prompt_constraints_language_coverage.format(
                        # params=json.dumps(params, indent=4),
                        # vars=json.dumps(vars, indent=4),
                        constraint=json.dumps(c["description"], indent=4),
                        constraint_new=json.dumps(code, indent=4),
                    )
                    res_check = get_response(coverage_check, model=model)
                    code_check = extract_code_from_end(res_check)
                    logger.log("\n\n++++++++++++")
                    logger.log(
                        "The LLM's response for checking the consistency of the two descriptions:")
                    logger.log(code_check)
                    logger.log("++++++++++++")

                    # check if the code is correct
                    if code_check == "YES":
                        c["error"] = "NO"
                        logger.log(
                            "The constraint consistency check is correct, we do not need to update the constraint anymore.")
                        break
                    else:
                        c["error"] = "YES"
                        update_constraints = True
                        logger.log(
                            "The constraint code is not correct, the two descriptions are not consistent, please regenerate the constraint code!")
                        break
                elif coverage == "math":
                    # check the constraint is correct
                    if is_semantically_consistent(c["description"], code):
                        c["error"] = "NO"
                        logger.log(
                            "The constraint consistency check is correct, we do not need to update the constraint anymore.")
                        break
                    else:
                        c["error"] = "YES"
                        update_constraints = True
                        logger.log(
                            "The constraint code is not correct, the two descriptions are not consistent, please regenerate the constraint code!")
                        break
                else:
                    raise ValueError(
                        "The consistency check mothod should be either prompt or math.")
            except:
                k -= 1
                logger.log(
                    "The LLM's response is not correct, please check the code!")
                if k == 0:
                    logger.log(
                        "Failed to generate the description for the constraint, please check the code!")
                    raise Exception("Failed to generate constraint code")

    logger.log("\n\n\n\n")
    logger.log("The generated descriptions for the constraints are:")
    logger.log("========================================")
    logger.log(json.dumps(languaged_constraints, indent=4))
    logger.log("========================================")
    logger.log(
        "We have generated the descriptions for the constraints, and whether the descriptions are correct or not.")
    logger.log(json.dumps(languaged_constraints, indent=4))

    if update_constraints:
        logger.log(
            "The constraints are not correct, we need to update the constraints.")
        logger.log(
            "The number of constraints that need to be updated is: " + str(error_count))
        logger.log("The constraints that need to be updated are:")
        for c in languaged_constraints:
            if c["error"] == "YES":
                logger.log(json.dumps(c, indent=4))
                logger.log("========================================")
                logger.log("The original description for the constraint is:")
                logger.log(c["description"])
                logger.log("========================================")
                logger.log("The code for the constraint is:")
                logger.log(c["code"])
                logger.log("========================================")
                logger.log("The new description for the constraint is:")
                logger.log(c["description_new"])
                logger.log("========================================")
    else:
        logger.log(
            "The constraints are correct, we do not need to update the constraints.")

    return languaged_constraints, update_constraints
