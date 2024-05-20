from transformers import TrainerCallback

class EvaluateAtStartCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        control.should_evaluate = True  # Set this flag to force evaluation at the beginning
