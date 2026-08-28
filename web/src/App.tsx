import WorkbenchBusyOverlay from "./components/workbench/WorkbenchBusyOverlay";
import WorkbenchContextRail from "./components/workbench/WorkbenchContextRail";
import WorkbenchErrorAlert from "./components/workbench/WorkbenchErrorAlert";
import WorkbenchHeader from "./components/workbench/WorkbenchHeader";
import WorkbenchLeftRail from "./components/workbench/WorkbenchLeftRail";
import WorkbenchStatusBar from "./components/workbench/WorkbenchStatusBar";
import { useWorkbenchShell } from "./components/workbench/useWorkbenchShell";
import { WorkbenchProvider, useWorkbench } from "./context/WorkbenchContext";
import TutorialGuide from "./tutorial/TutorialGuide";

function WorkbenchLayout() {
  const { error, setError, busy, dataset, result } = useWorkbench();
  const shell = useWorkbenchShell();
  const Page = shell.Page;

  return (
    <div className="app-root">
      <WorkbenchHeader
        progressMeta={shell.progressMeta}
        progressLabel={shell.progressLabel}
        progressPercent={shell.progressPercent}
        onTutorialRequest={shell.requestTutorial}
      />

      <main className={`app-shell ${shell.contextRailCollapsed ? "context-rail-collapsed" : ""}`}>
        <WorkbenchLeftRail
          mode={shell.mode}
          step={shell.step}
          visibleSteps={shell.visibleSteps}
          activeAuxiliaryPage={shell.activeAuxiliaryPage}
          experimentAvailable={shell.experimentAvailable}
          canOpenStep={shell.canOpenStep}
          workflowCompletion={shell.workflowCompletion}
          onOpenStep={shell.openStep}
          onOpenConversation={shell.openConversation}
          onOpenExperiment={shell.openExperiment}
        />

        <section className="content" data-tutorial="workspace">
          <div className="content-inner">
            <WorkbenchErrorAlert error={error} onClose={() => setError(null)} />
            <Page />
          </div>
        </section>

        <WorkbenchContextRail
          collapsed={shell.contextRailCollapsed}
          mode={shell.mode}
          activeAuxiliaryPage={shell.activeAuxiliaryPage}
          onToggle={shell.toggleContextRail}
        />
      </main>

      <WorkbenchStatusBar
        mode={shell.mode}
        activeAuxiliaryPage={shell.activeAuxiliaryPage}
      />

      <TutorialGuide
        requestId={shell.tutorialRequest}
        mode={shell.mode}
        hasDataset={Boolean(dataset)}
        hasResult={Boolean(result)}
      />

      <WorkbenchBusyOverlay busy={busy} />
    </div>
  );
}

export default function App() {
  return (
    <WorkbenchProvider>
      <WorkbenchLayout />
    </WorkbenchProvider>
  );
}
