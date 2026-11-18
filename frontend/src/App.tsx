import { Navigate, Route, Routes } from "react-router";
import AppShell from "./components/AppShell";
import { ENGINE_PATH, HOME } from "./lib/engines";
import ArtifactsPage from "./pages/ArtifactsPage";
import MtrlPage from "./pages/MtrlPage";
import PathsPage from "./pages/PathsPage";
import RunConsolePage from "./pages/RunConsolePage";
import RunsPage from "./pages/RunsPage";
import SessionsPage from "./pages/SessionsPage";
import SetupPage from "./pages/SetupPage";
import TrainingSessionsPage from "./pages/TrainingSessionsPage";

export default function AppRoutes() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<Navigate to={HOME} replace />} />
        <Route path="setup" element={<SetupPage />} />
        {/* One scenario page per engine (docs/ui-design.md §6.1); /scenarios was the page both shared. */}
        <Route path={ENGINE_PATH["training-session"]} element={<TrainingSessionsPage />} />
        <Route path={ENGINE_PATH.mtrl} element={<MtrlPage />} />
        <Route path="scenarios" element={<Navigate to={HOME} replace />} />
        <Route path="runs" element={<RunsPage />} />
        <Route path="runs/:id" element={<RunConsolePage />} />
        <Route path="artifacts" element={<ArtifactsPage />} />
        <Route path="capacity" element={<SessionsPage />} />
        <Route path="sessions" element={<Navigate to="/capacity" replace />} />
        <Route path="paths" element={<PathsPage />} />
        <Route path="*" element={<Navigate to={HOME} replace />} />
      </Route>
    </Routes>
  );
}
