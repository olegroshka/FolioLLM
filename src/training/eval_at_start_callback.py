from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments

class EvaluateAtStartCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if not state.is_world_process_zero:
            return  # Ensure this is only done by the main process
        print("Evaluating at the start of training...")
        self.trainer.evaluate()
        return control
