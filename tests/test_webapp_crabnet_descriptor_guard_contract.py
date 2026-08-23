from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CrabNetDescriptorGuardContractTest(unittest.TestCase):
    def test_run_settings_disable_derived_descriptors_for_crabnet_only(self) -> None:
        text = (ROOT / "web" / "src" / "context" / "useWorkbenchRunSettings.ts").read_text(
            encoding="utf-8"
        )

        self.assertIn('import { isCrabNetModelType } from "../modelOptions";', text)
        self.assertIn("effectiveCompositionSettings", text)
        self.assertIn("includeDescriptors: false", text)
        self.assertIn("compositionSettings: effectiveCompositionSettings", text)

    def test_composition_ui_explains_crabnet_descriptor_behavior(self) -> None:
        text = (ROOT / "web" / "src" / "components" / "CompositionModelSettings.tsx").read_text(
            encoding="utf-8"
        )

        self.assertIn("descriptorDisabledForModel = isCrabNetModelType(modelType)", text)
        self.assertIn("CrabNetでは派生記述子を追加しません", text)
        self.assertIn("disabled={descriptorDisabledForModel}", text)
        self.assertIn("settings.includeDescriptors && !descriptorDisabledForModel", text)


if __name__ == "__main__":
    unittest.main()
