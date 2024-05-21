from transformers import TrainerCallback

class EvaluateAtStartCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return  # Ensure this is only done by the main process
        print("Evaluating at the start of training...")
        control.should_evaluate = True  # Force evaluation at the beginning
        return control
