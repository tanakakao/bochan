import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { installCompositionDatasetState } from "./compositionDatasetState";
import { installCompositionExtension } from "./compositionExtension";
import { installCompositionPrepareControls } from "./compositionPrepareControls";
import { installCompositionVisualizationGuard } from "./compositionVisualizationGuard";
import { installDynamicNumberInputSteps } from "./numericInputUtils";
import { installWorkflowLayoutExtension } from "./workflowLayoutExtension";
import "./styles.css";
import "./target-settings.css";
import "./constraint-settings.css";
import "./advanced-settings.css";
import "./constraint-selection.css";
import "./workflow-separation.css";
import "./ui-adjustments.css";
import "./model-artifact.css";
import "./result-interactions.css";
import "./experiment-results.css";
import "./experiment-history.css";
import "./styles/workbench-design.css";
import "./data-dropzone.css";
import "./composition-extension.css";
import "./workflow-layout-extension.css";
import "./red-theme.css";

installDynamicNumberInputSteps();
installCompositionPrepareControls(installCompositionExtension);
installCompositionDatasetState();
installCompositionVisualizationGuard();
installWorkflowLayoutExtension();

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);