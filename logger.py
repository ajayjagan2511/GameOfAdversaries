import json
from pathlib import Path
import wandb


class Logger:
    def __init__(self, log_path: str = "attacks.log", project: str = "adversarial_sds_attack"):
        self.log_file = Path(log_path)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.run = wandb.init(project=project, reinit=True)

    def log(self, record: dict):
        with self.log_file.open("a") as f:
            f.write(json.dumps(record) + "\n")
        self.run.log({k: v for k, v in record.items() if isinstance(v, (int, float))})
        self.run.log({"attack_success": record.get("attack_success", False)})
        self.run.log({"cosine_similarity": record.get("cosine_similarity", 0.0)})
        self.run.log({"original_harmscore": record.get("original_harmscore", 0.0), "perturbed_harmscore": record.get("perturbed_harmscore", 0.0)}) 