from pathlib import Path

path = Path("src/bochan/api/optimizer/__init__.py")
text = path.read_text(encoding="utf-8")

old = '''        direct_known_noise = False
        if train_Yvar is not None and observation_data is None:
            import torch

            y_tensor = torch.as_tensor(train_Y)
            yvar_tensor = torch.as_tensor(train_Yvar)
'''
new = '''        direct_known_noise = False
        if train_Yvar is not None and observation_data is None:
            if train_X is None or train_Y is None:
                raise ValueError("Provide both train_X and train_Y with train_Yvar.")

            import torch

            y_tensor = torch.as_tensor(train_Y)
            yvar_tensor = torch.as_tensor(train_Yvar)
'''
if old not in text:
    raise SystemExit("direct known-noise block not found")
text = text.replace(old, new, 1)

old = '''        if direct_known_noise:
            if train_X is None or train_Y is None:
                raise ValueError("Provide both train_X and train_Y with train_Yvar.")

            base_model_config = model_config or self.model_config
'''
new = '''        if direct_known_noise:
            base_model_config = model_config or self.model_config
'''
if old not in text:
    raise SystemExit("redundant direct known-noise guard not found")
text = text.replace(old, new, 1)

old = '''        self.bundle = build_objective_bundle(
            train_X=objective_X,
            train_Y=objective_Y,
            train_Yvar=objective_Yvar,
            config=self.model_config,
            model_registry=self.model_registry,
        )
'''
new = '''        if objective_Yvar is None:
            self.bundle = build_objective_bundle(
                train_X=objective_X,
                train_Y=objective_Y,
                config=self.model_config,
                model_registry=self.model_registry,
            )
        else:
            self.bundle = build_objective_bundle(
                train_X=objective_X,
                train_Y=objective_Y,
                train_Yvar=objective_Yvar,
                config=self.model_config,
                model_registry=self.model_registry,
            )
'''
if old not in text:
    raise SystemExit("objective bundle block not found")
text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")
