import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import DynamicNumberStepBridge from "./components/DynamicNumberStepBridge";
import WorkbenchPageErrorBoundary from "./components/WorkbenchPageErrorBoundary";
import { installCompositionDatasetState } from "./compositionDatasetState";
import { installCompositionRuntime } from "./compositionRuntime";
import "./styles.css";
import "./styles/typography.css";
import "./target-settings.css";
import "./advanced-settings.css";
import "./styles/workflow.css";
import "./model-artifact.css";
import "./result-interactions.css";
import "./experiment-results.css";
import "./experiment-history.css";
import "./styles/workbench-design.css";
import "./composition-extension.css";
import "./red-theme.css";

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
