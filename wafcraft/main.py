import argparse
import os

from config import (
    Surrogate_Data_V1_Config,
    Surrogate_SVM_V1_Config,
    Target_Config,
    Test_Config,
    Test_Surrogate_Overlap_V1_Config,
    Test_Surrogate_SVM_V1_Config,
)
from src.pipeline import run_model_pipeline
from src.utils import (
    generate_workspace_path,
    get_config_string,
    get_most_recent_data_path,
    log,
    save_settings,
)


if __name__ == "__main__":
    most_recent_data_path = get_most_recent_data_path()

    parser = argparse.ArgumentParser(description="Run config")  # TODO

    parser.add_argument(
        "--config",
        type=str,
        help="Choose the configuration to use.",
        required=True,
    )

    # require either --workspace "path_to_workspace" or --new
    workspace_group = parser.add_mutually_exclusive_group(required=True)
    workspace_group.add_argument(
        "--workspace",
        type=str,
        help="Directory name of the workspace",
    )
    workspace_group.add_argument(
        "--new",
        action="store_true",
        help="Create a new workspace",
    )

    args = parser.parse_args()

    # Set config based on the argument
    if args.config == "test":
        Config = Test_Config
    elif args.config == "test_surrogate_overlap_v1":
        Config = Test_Surrogate_Overlap_V1_Config
    elif args.config == "test_surrogate_svm_v1":
        Config = Test_Surrogate_SVM_V1_Config
    elif args.config == "target":
        Config = Target_Config
    elif args.config == "surrogate_svm_v1":
        Config = Surrogate_SVM_V1_Config
    elif args.config == "surrogate_data_v1":
        Config = Surrogate_Data_V1_Config
    else:
        raise ValueError("Invalid config")

    if args.new:
        workspace = generate_workspace_path()
        os.makedirs(workspace, exist_ok=True)
        save_settings(Config, workspace)
        log(f'\n\n\n{"-"*60}', 2)
        log("starting new workspace", 2)
        log(f"workspace: {workspace}", 2)
        log(f"using config:\n{get_config_string(Config)}", 2)
    else:
        workspace = f"/app/wafcraft/data/prepared/{args.workspace}"
        if not os.path.exists(workspace):
            raise ValueError(f"Workspace {workspace} does not exist")

    run_model_pipeline(Config, workspace)
