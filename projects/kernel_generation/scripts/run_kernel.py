from __future__ import annotations
import os
import sys
import torch
from problem_loader import load_module, pick_device, solution_class, synchronize

def main() -> None:
    cfg_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    device = pick_device()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prob = load_module(os.path.join(script_dir, 'problem.py'), 'problem')
    sol_module = load_module(os.path.join('solution', 'solution.py'), 'solution')
    sol_cls = solution_class(sol_module)
    configs = list(prob.make_configs())
    if not 0 <= cfg_index < len(configs):
        raise SystemExit(f'config_index {cfg_index} out of range (0..{len(configs) - 1})')
    cfg = configs[cfg_index]
    model = sol_cls(*cfg.init_args, **cfg.init_kwargs)
    if isinstance(model, torch.nn.Module):
        model = model.to(device).eval()
        ref = prob.Reference(*cfg.init_args, **cfg.init_kwargs).to(device).eval()
        model.load_state_dict(ref.state_dict())
    inputs = cfg.make_inputs()
    with torch.no_grad():
        for _ in range(iters):
            model(*inputs)
    synchronize(device)
    print(f'ran {sol_cls.__name__} on {cfg.name} x{iters} on {device}')
if __name__ == '__main__':
    main()
