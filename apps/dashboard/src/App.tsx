import { BaselineView } from "./routes/baseline";
import "./App.css";

/**
 * The four module views share one time axis so a single replayed episode reads across all of them:
 * vulnerability map (M1) -> uncertainty (M2) -> gating timeline (M3) -> verification log (M4).
 * Keeping the axis shared is what makes the dashboard an explanation rather than four widgets.
 *
 * Those four are not built yet. What is built is the measured baseline they will be judged
 * against — the CEB ledger, served from the Module 1 calibration pipeline.
 */
export default function App() {
  return (
    <div className="app">
      <header className="app__header">
        <div className="app__wrap app__bar">
          <div className="app__id">
            <span className="app__mark">MetaCore</span>
            <span className="app__project">J26-DS-317</span>
          </div>
          <nav className="app__nav" aria-label="Views">
            <span className="app__tab is-on" aria-current="page">
              Baseline
            </span>
            <span className="app__tab is-off" title="Not built yet">State</span>
            <span className="app__tab is-off" title="Not built yet">Uncertainty</span>
            <span className="app__tab is-off" title="Not built yet">Gating</span>
            <span className="app__tab is-off" title="Not built yet">Verification</span>
          </nav>
        </div>
      </header>

      <main className="app__wrap app__main">
        <div className="app__intro">
          <h1 className="app__title">Four islanded microgrids, as measured</h1>
          <p className="app__lede">
            Analaitivu, Eluvaitivu, Nainativu and Neduntivu run without a mainland connection, on
            diesel barged in by boat. There is no SCADA and no historian on any of them — this is
            every number that exists, and it arrives monthly.
          </p>
        </div>
        <BaselineView />
      </main>
    </div>
  );
}
