import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import DynamicNumberStepBridge from "./components/DynamicNumberStepBridge";
import WorkbenchPageErrorBoundary from "./components/WorkbenchPageErrorBoundary";
import { installCompositionDatasetState } from "./compositionDatasetState";
import { installCompositionRuntime } from "./compositionRuntime";
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
import "./readability.css";
import "./ux-simplification.css";
import "./ux-enhancements.css";
import "./ux-corrections.css";

installCompositionRuntime();
installCompositionDatasetState();

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <WorkbenchPageErrorBoundary resetKey="workbench-root">
      <DynamicNumberStepBridge />
      <App />
    </WorkbenchPageErrorBoundary>
  </React.StrictMode>
);