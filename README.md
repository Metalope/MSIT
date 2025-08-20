# MSIT

A multi-stage training pipeline built on the custom **Find&Kill** environment.

## Overview

**MSIT** provides a modular, multi-phase training workflow leveraging a custom environment (`env_new/`) and standardized interface (`env.py`). Each training phase—from initial detection to final policy refinement—is encapsulated in its own script, ensuring clarity and flexibility.


## Module Details

### `env.py`

Defines the interface for the **Find&Kill** environment, typically including methods like:

- `reset()`: Initialize the environment and return starting state.
- `step(action)`: Apply an action, return `(next_state, reward, done, info)`.
- `render()`: Optional—visualize the environment.

This modular design ensures training scripts can interact with the environment without knowledge of its internal workings.

### `env_new/`

Contains the full implementation of the **Find&Kill** environment, including:

- State representation
- Action definitions
- Reward functions
- Environment dynamics and logic

You can extend or modify environment behavior here as required.

## Training Phases

The training process is divided into discrete phases, each with a dedicated script:

1. **`detector_train.py`**  
   Trains an initial detector—useful for identifying targets.

   ```bash
   python detector_train.py

2. **`phase1_train.py`**
   Conducts Phase 1 training—often used for initializing policies or exploration strategies.

   ```bash
   python phase1_train.py
   ```

3. **`phase2_train.py`**
   Refines the model/policy starting from Phase 1 outputs.

   ```bash
   python phase2_train.py
   ```

4. **`phase3_train.py`**
   Final training stage—fine-tunes performance for evaluation or deployment.

   ```bash
   python phase3_train.py
   ```
### License

Specify your project’s license here (e.g., MIT, Apache-2.0).
