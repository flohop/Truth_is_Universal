# intervention.py

import torch as t


class InterventionHook:
    """
    A hook that can be used to read or modify activations during a forward pass.
    This version is robust to different activation tensor shapes and devices.
    """

    def __init__(self):
        self.out = None
        self.intervention_vector = None
        self.alpha = 1.0

    def __call__(self, module, module_inputs, module_outputs):
        original_activation = module_outputs[0]
        self.out = original_activation.clone()

        if self.intervention_vector is not None:
            # Check if the activation tensor has a batch dimension
            if original_activation.ndim == 3:  # Shape is [batch, sequence, hidden_dim]
                final_token_activation = original_activation[0, -1, :]
                modified_activation = final_token_activation + (self.alpha * self.intervention_vector)
                module_outputs[0][0, -1, :] = modified_activation

            elif original_activation.ndim == 2:  # Shape is [sequence, hidden_dim]
                final_token_activation = original_activation[-1, :]
                modified_activation = final_token_activation + (self.alpha * self.intervention_vector)
                module_outputs[0][-1, :] = modified_activation
            else:
                raise ValueError(f"Unexpected activation shape: {original_activation.shape}")

    def set_intervention(self, vector, alpha=1.0):
        """
        Set the vector and strength for the next intervention.
        Ensure the intervention vector is on the same device as the activation tensor will be.
        """
        # The activation tensors live on the module's device, so the vector should too.
        # We can get the device from any parameter of the module the hook is attached to.
        # However, a simpler and robust way is to just move it to the correct device when we
        # run the experiment, as we know the model's device there.
        # For now, let's just make sure it's moved when set. The run_experiment function will handle device.
        self.intervention_vector = vector  # Keep it on its current device for now
        self.alpha = alpha

    def reset(self):
        """Reset the hook to a passive, observational state."""
        self.intervention_vector = None


# --- FIX IS IN THIS FUNCTION ---
def run_intervention_experiment(model, tokenizer, shortcut_vector, test_statement, target_layer, alpha):
    """
    Runs a statement through the model with and without intervention and prints the results.
    """
    device = model.device  # This will be 'mps:0' on your Mac

    intervention_hook = InterventionHook()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hook_handle = model.model.layers[target_layer].register_forward_hook(intervention_hook)

    print("-" * 60)
    print(f"🔬 Experiment: '{test_statement}'")
    print(f"   Intervening at Layer {target_layer} with alpha = {alpha}")

    inputs = tokenizer(test_statement, return_tensors="pt", padding=True).to(device)

    # 1. Baseline Run
    intervention_hook.reset()
    baseline_output_ids = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    baseline_completion = tokenizer.decode(baseline_output_ids[0], skip_special_tokens=True)

    print(f"\n✅ Baseline (Normal Run):")
    print(f"   '{baseline_completion}'")

    # 2. Intervention Run
    # Explicitly move the shortcut_vector to the model's device before setting it in the hook
    intervention_hook.set_intervention(shortcut_vector.to(device), alpha=alpha)

    intervention_output_ids = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    intervention_completion = tokenizer.decode(intervention_output_ids[0], skip_special_tokens=True)

    print(f"\n🧠 Intervention Run:")
    print(f"   '{intervention_completion}'")

    hook_handle.remove()
    print("-" * 60)