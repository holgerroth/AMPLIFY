
from mlp import MLP

from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob
from nvflare.job_config.script_runner import ScriptRunner

if __name__ == "__main__":
    n_clients = 2
    num_rounds = 10
    train_script = "train_mlp_fl.py"

    # Create BaseFedJob with initial model
    job = BaseFedJob(
      name="bindings_mlp",
      initial_model=MLP(input_dim=640),
    )

    # Define the controller and send to server
    controller = FedAvg(
        num_clients=n_clients,
        num_rounds=num_rounds,
    )
    job.to_server(controller)

    # Add clients
    for i in range(n_clients):
        runner = ScriptRunner(script=train_script)
        job.to(runner, f"site-{i}")

    job.export_job("job_config")
    job.simulator_run("/tmp/nvflare/amplify", gpu="0")
