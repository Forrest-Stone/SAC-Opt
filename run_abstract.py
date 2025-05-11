import argparse
import time
import os
import json

from util import *
from parameters import get_params
from constraint import get_constraints
from objective import get_objective
from variables import get_vars
from model2data import generate_data
from data2model import generate_model
from generate_code import generate_code
from execute_code import execute_and_debug
from update_util import update_constraints


def convert_parameters(original_parameters):
    new_parameters = []
    for key, value in original_parameters.items():
        new_parameters.append({
            "definition": value["definition"],
            "symbol": key,
            "value": "",
            "shape": value["shape"]
        })
    return new_parameters


def main():
    parser = argparse.ArgumentParser(
        description="Run the algorithm on the dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        help='Dataset name, "nl4opt" or "ComplexOR" or "nlp4lp", "IndustryOR"， or "complexlp", or "easylp", or "resocratic"',
        default="complexor",
    )
    parser.add_argument("--problem", type=str,
                        help="Problem name", default="1")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Base large language model",
    )
    parser.add_argument("--run_dir", type=str, help="Run directory")
    parser.add_argument(
        "--coverage",
        type=str,
        default="prompt",
        help="The method to use for coverage, prompt or math",
    )

    args = parser.parse_args()

    if not args.model in [
        "gpt-4-1106-preview",
        "gpt-3.5-turbo",
        "gpt-4o",
    ]:
        print(
            "Invalid model name! Please choose from 'gpt-4-1106-preview', 'gpt-3.5-turbo'"
        )
        exit(0)

    dataset = args.dataset.lower()
    problem = args.problem
    llm_model = args.model
    ERROR_CORRECTION = True

    # dataser path
    dir = os.path.join("data", args.dataset)

    if not os.path.exists(dir):
        print(f"Dataset {args.dataset} not found!")
        exit(0)
    if not os.path.exists(os.path.join(dir, problem)):
        print(f"Problem {problem} not found!")
        exit(0)

    # create the log directory
    run_dir = args.run_dir
    print(run_dir)
    if run_dir is None:
        run_dir = f"{llm_model}/{dataset}/{problem}/run_{time.strftime('%Y%m%d%H%M%S')}_{dataset}_{problem}"
        log_dir = f"{run_dir}/logs"

    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    print(run_dir)
    print(log_dir)

    logger = Logger(f"{log_dir}/log.txt")
    logger.reset()

    print(f"Solving {args.dataset} problem {problem}...")

    logger.log("Run directory: " + run_dir)
    logger.log(f"Solving {args.dataset} problem {problem}...")

    # read the problem description
    with open(os.path.join(dir, problem, "description.txt"), "r") as f:
        description = f.read()

    ## read the data json for abstract model
    abs_data_dir = f"data/{args.dataset}/{problem}/sample.json"
    if not os.path.exists(abs_data_dir):
        print(f"Data file {abs_data_dir} not found!")
        exit(0)
    with open(abs_data_dir, "r") as f:
        data = json.load(f)

    print(data)

    # save the answer
    answer = data[0]["output"][0]
    with open(os.path.join(dir, problem, "answer.txt"), "w") as f:
        f.write(f"{answer}")

    # read the input parameters for the abstract model
    origin_parameters_data = data[0]["input"]
    origin_parameters_data = json.dumps(origin_parameters_data, indent=4)

    # get the parameters
    parameters = get_params(description, origin_parameters_data, check=ERROR_CORRECTION)
    # save the parameters
    with open(os.path.join(dir, problem, "params.json"), "w") as f:
        json.dump(parameters, f, indent=4)

    # create the json state containing the problem description and parameters
    state = create_state(os.path.join(dir, problem),
                         os.path.join(dir, problem))
    print(state)
    save_state(state, os.path.join(dir, problem, "state_1_params.json"))

    state = load_state(os.path.join(dir, problem, "state_1_params.json"))
    vars = get_vars(
        state["description"],
        state["parameters"],
        vars=None,
        check=ERROR_CORRECTION,
    )
    print(vars)
    state["variables"] = vars
    save_state(state, os.path.join(dir, problem, "state_2_variables.json"))

    # Get objective
    state = load_state(os.path.join(dir, problem, "state_2_variables.json"))
    objective = get_objective(
        state["description"],
        state["parameters"],
        state["variables"],
        check=ERROR_CORRECTION,
        logger=logger,
        model=llm_model,
    )
    print(objective)
    state["objective"] = objective
    save_state(state, os.path.join(dir, problem, "state_3_objective.json"))

    # Get constraints
    state = load_state(os.path.join(dir, problem, "state_3_objective.json"))
    constraints = get_constraints(
        state["description"],
        state["parameters"],
        state["variables"],
        check=ERROR_CORRECTION,
        logger=logger,
        model=llm_model,
    )
    print(constraints)
    state["constraints"] = constraints
    save_state(state, os.path.join(dir, problem, "state_4_constraints.json"))

    ## change parameter state
    state["parameters"] = convert_parameters(state["parameters"])

    # add flags for the constraints
    for constraint in state["constraints"]:
        constraint["error"] = ""

    # add flags for the objective
    state["objective"]["error"] = ""
    save_state(state, os.path.join(dir, problem, "input_targets.json"))

    data_dir = f"/datapath/data/{args.dataset}/{problem}/data.json"
    if not os.path.exists(data_dir):
        print(f"Data file {data_dir} not found!")
        exit(0)

    with open(f"data/{args.dataset}/{problem}/input_targets.json", "r") as f:
        state = json.load(f)

    save_state(state, os.path.join(run_dir, "state_0.json"))

    state = prep_problem_json(state)

    state = {
        "description": state["description"],
        "parameters": state["parameters"],
        "constraints": state["constraints"],
        "variables": state["variables"],
        "objective": state["objective"],
        "log_folder": log_dir,
        "data_json_path": f"data/{dataset}/{problem}/data.json",
    }
    if not os.path.exists(state["log_folder"]):
        os.makedirs(state["log_folder"])

    sanity_check(state)
    # print(state)

    # save the initial state
    save_state(state, os.path.join(run_dir, "state_init.json"))

    logger.log("Run directory: " + run_dir)
    logger.log(f"Solving {args.dataset} problem {problem}...")

    t0 = time.time()
    # the first state: data2model
    state = load_state(os.path.join(run_dir, "state_init.json"))
    logger.log("Initial state:")
    logger.log(json.dumps(state, indent=4))

    constriants, objective = generate_model(
        state["description"],
        state["parameters"],
        state["variables"],
        state["constraints"],
        state["objective"],
        model=llm_model,
        logger=logger,
    )

    state["constraints"] = constriants
    state["objective"] = objective
    save_state(state, os.path.join(run_dir, "state_code_0.json"))

    # self correction loop
    iteration_count = 0
    MAX_ITER = 5

    # the second state: model2data
    state = load_state(os.path.join(run_dir, "state_code_0.json"))
    print("++++++++++++")
    print("Initial state:")
    print(state)
    print("++++++++++++")

    while iteration_count < MAX_ITER:
        print(f"Iteration {iteration_count + 1}...")

        # load the state
        state_file = f"state_code_{iteration_count}.json"
        state = load_state(os.path.join(run_dir, state_file))

        # generate the data and judge the difference
        constriants_new, update_need = generate_data(
            state["description"],
            state["parameters"],
            state["variables"],
            state["constraints"],
            model=llm_model,
            logger=logger,
            coverage=args.coverage,
        )

        state["constraints"] = constriants_new
        print(state["constraints"])
        # save_state(state, os.path.join(run_dir, "state_data.json"))

        if update_need:
            print("Constraints need to be updated...")
            # update the error constraints
            constraints_updated = update_constraints(
                state["description"],
                state["parameters"],
                state["variables"],
                state["constraints"],
                model=llm_model,
                logger=logger,
            )
            state["constraints"] = constraints_updated
            print("Constraints updated:")
            print(constraints_updated)
            save_state(state, os.path.join(
                run_dir, f"state_code_{iteration_count + 1}.json"))
            if iteration_count == MAX_ITER - 1:
                print("Maximum tries reached. Save the last state.")
                save_state(state, os.path.join(run_dir, "state_code.json"))
        else:
            print("Constraints do not need to be updated...")
            # save the state
            save_state(state, os.path.join(run_dir, "state_code.json"))
            break

        iteration_count += 1

    if iteration_count >= MAX_ITER:
        print("Maximum iterations reached. Exiting...")

    # generate the code and run
    state = load_state(os.path.join(run_dir, "state_code.json"))
    generate_code(state, run_dir, data_dir)
    execute_and_debug(state, model=llm_model, dir=run_dir, logger=logger)
    #######

    t1 = time.time()
    # Calculate time taken
    time_taken = t1 - t0
    print(f"Time taken: %.4f seconds" % time_taken)

    # Write time taken to time.txt
    time_file_path = f"data/{dataset}/{problem}/time.txt"
    with open(time_file_path, "w") as f:
        f.write(f"{time_taken:.4f}")

    print("DONE!")


if __name__ == "__main__":
    main()
