# Video Style Transfer With Reinforcement Learning
 **Acknowledgement:** The codebase is build on [DiffuseST](https://github.com/I2-Multimedia-Lab/DiffuseST/tree/main) to extract latent representation for each frame and perform encoding and decoding stage of the diffusion model. All code for the policy gradient training loop, the policy network architecture, reward calculation, loss functions, and analysis is ours.

The main implementation is located in the `DiffuseST_rl` directory.

### Training
To train the model, run:

```bash
python train.py [your arguments here]
```

### Evaluation
To evaluate the stylized output images, run:
```bash
python eval.py [your arguments here]
```
### Baseline
To try the baseline method, run:
```bash
python run_baseline.py [your arguments here]
```
 
