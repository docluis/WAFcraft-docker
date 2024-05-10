import argparse
import os

from config import (
    Large_Surrogate_Data_V1_Config,
    Large_Surrogate_Data_V4_Config,
    Large_Surrogate_SVM_V1_Config,
    Large_Target_Config,
    NoAdv_Surrogate_Data_V1_Config,
    NoAdv_Surrogate_Data_V2_Config,
    NoAdv_Surrogate_Data_V3_Config,
    NoAdv_Surrogate_Data_V4_Config,
    NoAdv_Surrogate_Data_V5_Config,
    NoAdv_Target_Config,
    Surrogate_Data_V1_Config,
    Surrogate_Data_V2_Config,
    Surrogate_Data_V3_Config,
    Surrogate_Data_V4_Config,
    Surrogate_Data_V5_Config,
    Surrogate_Paranoia_V1_Config,
    Surrogate_Paranoia_V2_Config,
    Surrogate_Paranoia_V3_Config,
    Surrogate_SVM_V1_Config,
    Surrogate_SVM_V2_Config,
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
from src.transfer import test_transferability


if __name__ == "__main__":
    most_recent_data_path = get_most_recent_data_path()

    parser = argparse.ArgumentParser(description="Run config")  # TODO

    parser.add_argument(
        "--config",
        type=str,
        help="Choose the configuration to use.",
        required=True,
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--data",
        action="store_true",
        help="Generate data for a model",
    )
    mode_group.add_argument(
        "--transfer",
        action="store_true",
        help="Test transfer of samples between models",
    )

    # Data arguments
    data_group = parser.add_mutually_exclusive_group(required=False)
    data_group.add_argument(
        "--workspace",
        type=str,
        help="Directory name of the workspace",
    )
    data_group.add_argument(
        "--new",
        action="store_true",
        help="Create a new workspace",
    )

    # transfer arguments
    parser.add_argument(
        "--target",
        type=str,
        help="Directory name of the target workspace",
    )
    parser.add_argument(
        "--surrogate",
        type=str,
        help="Directory name of the surrogate workspace",
    )
    parser.add_argument(
        "--samples",
        type=int,
        help="Number of samples to test",
        default=-1,  # -1 means all samples
    )
    parser.add_argument(
        "--target_use_adv",
        type=bool,
        help="Use adversarial model for target (default: True)",
        default=True,
    )
    parser.add_argument(
        "--surrogate_use_adv",
        type=bool,
        help="Use adversarial model for surrogate (only important for output) (default: True)",
        default=True,
    )

    args = parser.parse_args()

    # Check if the arguments are valid
    if args.data and not (args.workspace or args.new):
        parser.error("--data requires either --workspace or --new.")
    if args.transfer and not (args.target and args.surrogate):
        parser.error("--transfer requires --target and --surrogate.")

    # Set config based on the argument
    Configs = [
        Target_Config,
        Surrogate_Data_V1_Config,
        Surrogate_Data_V2_Config,
        Surrogate_Data_V3_Config,
        Surrogate_Data_V4_Config,
        Surrogate_Data_V5_Config,
        Surrogate_SVM_V1_Config,
        Surrogate_SVM_V2_Config,
        Surrogate_Paranoia_V1_Config,
        Surrogate_Paranoia_V2_Config,
        Surrogate_Paranoia_V3_Config,
        Test_Config,
        Test_Surrogate_Overlap_V1_Config,
        Test_Surrogate_SVM_V1_Config,
        Large_Target_Config,
        Large_Surrogate_SVM_V1_Config,
        Large_Surrogate_Data_V1_Config,
        Large_Surrogate_Data_V4_Config,
        NoAdv_Target_Config,
        NoAdv_Surrogate_Data_V1_Config,
        NoAdv_Surrogate_Data_V2_Config,
        NoAdv_Surrogate_Data_V3_Config,
        NoAdv_Surrogate_Data_V4_Config,
        NoAdv_Surrogate_Data_V5_Config,
    ]

    Config = None
    for C in Configs:
        if C.NAME == args.config:
            Config = C
            break
    if Config is None:
        raise ValueError("Invalid config")

    if args.data:
        log(f'\n\n\n{"-"*60}', 2)
        log("[main] running model pipeline to generate new data...", 2)
        if args.new:
            workspace = generate_workspace_path()
            os.makedirs(workspace, exist_ok=True)
            save_settings(Config, workspace)
            log(f"starting new workspace: {workspace}", 2)
        else:
            workspace = f"/app/wafcraft/data/prepared/{args.workspace}"
            if not os.path.exists(workspace):
                raise ValueError(f"Workspace {workspace} does not exist")
            log(f"using existing workspace: {workspace}", 2)
            log(f"using config:\n{get_config_string(Config)}", 2)
        run_model_pipeline(Config, workspace)

    if args.transfer:
        log(f'\n\n\n{"-"*60}', 2)
        log("[main] testing transferability...", 2)
        target_workspace = f"/app/wafcraft/data/prepared/{args.target}"
        if not os.path.exists(target_workspace):
            raise ValueError(f"Workspace {target_workspace} does not exist")
        surrogate_workspace = f"/app/wafcraft/data/prepared/{args.surrogate}"
        if not os.path.exists(surrogate_workspace):
            raise ValueError(f"Workspace {surrogate_workspace} does not exist")
        test_transferability(
            Config,
            target_workspace,
            surrogate_workspace,
            args.samples,
            args.target_use_adv,
            args.surrogate_use_adv,
        )
    log("\n\n", 2)
