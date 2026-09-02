from pathlib import Path

path = Path("src/bochan/api/observation/state.py")
text = path.read_text(encoding="utf-8")
old = '''    def report(self) -> dict[str, Any]:
        return {
            "n_rows": int(self.X.shape[0]),
            "n_completed": int(self.completed_mask.sum().item()),
            "n_success": int(self.success_mask.sum().item()),
            "n_failed": int(self.failed_mask.sum().item()),
            "n_pending": int(self.pending_mask.sum().item()),
            "known_observation_variance": self.Yvar is not None,
            "observed_per_output": [
                int(value)
                for value in self.observed_mask.sum(dim=0).detach().cpu().tolist()
            ],
        }
'''
new = '''    def report(self) -> dict[str, Any]:
        report = {
            "n_rows": int(self.X.shape[0]),
            "n_completed": int(self.completed_mask.sum().item()),
            "n_success": int(self.success_mask.sum().item()),
            "n_failed": int(self.failed_mask.sum().item()),
            "n_pending": int(self.pending_mask.sum().item()),
            "observed_per_output": [
                int(value)
                for value in self.observed_mask.sum(dim=0).detach().cpu().tolist()
            ],
        }
        if self.Yvar is not None:
            report["known_observation_variance"] = True
        return report
'''
if old not in text:
    raise SystemExit("expected ObservationData.report block not found")
path.write_text(text.replace(old, new), encoding="utf-8")
