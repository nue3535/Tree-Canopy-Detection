import { useCallback, useEffect, useMemo, useState } from "react";

const TRAINING_STATUS_POLL_MS = 2500;

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

/** Row order and display names aligned with Assessment 3 comparison table. */
const EVALUATION_TABLE_ROWS = [
  { key: "unet", label: "U-Net" },
  { key: "deeplabv3plus", label: "DeepLabV3" },
  { key: "maskrcnn", label: "Mask R-CNN" },
  { key: "segformer", label: "SegFormer" },
  { key: "sam2", label: "SAM2 (Zero-shot)" }
];

const METHOD_LABELS = Object.fromEntries(EVALUATION_TABLE_ROWS.map(({ key, label }) => [key, label]));

function App() {
  const getInitialTheme = () => {
    if (typeof window === "undefined") return "blue";
    const saved = window.localStorage.getItem("ui_theme");
    return saved === "blue" || saved === "eco" || saved === "dark" ? saved : "blue";
  };

  const [activeView, setActiveView] = useState("evaluation");
  const [file, setFile] = useState(null);
  const [method, setMethod] = useState("unet");
  const [previewUrl, setPreviewUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [evaluationError, setEvaluationError] = useState("");
  const [trainingMethod, setTrainingMethod] = useState("unet");
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [trainingBusy, setTrainingBusy] = useState(false);
  const [backendApiStatus, setBackendApiStatus] = useState("checking");
  const [frontendStatus, setFrontendStatus] = useState("online");
  const [uiTheme, setUiTheme] = useState(getInitialTheme);
  const [trainingResults, setTrainingResults] = useState(null);
  const [trainingResultsLoading, setTrainingResultsLoading] = useState(false);

  const legendItems = useMemo(() => {
    if (!result?.class_labels || !result?.class_colors) return [];
    return Object.keys(result.class_labels)
      .sort((a, b) => Number(a) - Number(b))
      .map((classId) => ({
        classId,
        label: result.class_labels[classId],
        color: result.class_colors[classId] || "#000000"
      }));
  }, [result]);

  const isSelectedTrainingRunning = useMemo(() => {
    return trainingStatus?.status === "running" && trainingStatus?.method === trainingMethod;
  }, [trainingStatus, trainingMethod]);

  const trainingTierNote = useMemo(() => {
    const byMethod = {
      deeplabv3plus: "Training uses the best-quality preset (longer schedule, stronger expected accuracy). Prefer a capable GPU.",
      sam2: "Training uses the best-quality SAM2 preset (more steps, lower LR). Expect long runs and high memory use.",
      unet: "Training uses the best-quality U-Net preset (more epochs).",
      maskrcnn: "Training uses the best-quality Mask R-CNN preset; this path is the heaviest—GPU strongly recommended.",
      segformer: "Training uses the best-quality SegFormer preset (more epochs)."
    };
    return byMethod[trainingMethod] || byMethod.unet;
  }, [trainingMethod]);

  const onFileChange = (event) => {
    const selected = event.target.files?.[0] || null;
    setFile(selected);
    setResult(null);
    setError("");
    if (selected) {
      setPreviewUrl(URL.createObjectURL(selected));
    } else {
      setPreviewUrl("");
    }
  };

  const submit = async (event) => {
    event.preventDefault();
    if (!file) {
      setError("Please choose an image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("method", method);
      formData.append("strict_conservation_mode", "false");
      const response = await fetch(`${API_BASE}/api/segment`, {
        method: "POST",
        body: formData
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Segmentation request failed.");
      }
      setResult(payload);
    } catch (err) {
      setError(err.message || "Unexpected error");
    } finally {
      setLoading(false);
    }
  };

  const checkBackendHealth = async () => {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 4000);
    try {
      const response = await fetch(`${API_BASE}/api/health`, { signal: controller.signal });
      const payload = await response.json();
      if (response.ok && payload?.status === "ok") {
        setBackendApiStatus("online");
      } else {
        setBackendApiStatus("offline");
      }
    } catch {
      setBackendApiStatus("offline");
    } finally {
      clearTimeout(timeout);
    }
  };

  const loadTrainingResults = useCallback(async () => {
    setTrainingResultsLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/evaluation/training-results`);
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Failed to fetch training results.");
      }
      setTrainingResults(payload.results);
    } catch {
      setTrainingResults(null);
    } finally {
      setTrainingResultsLoading(false);
    }
  }, []);

  const loadTrainingStatus = useCallback(
    async (options = {}) => {
      const silent = Boolean(options.silent);
      if (!silent) {
        setTrainingBusy(true);
        setEvaluationError("");
      }
      try {
        const response = await fetch(`${API_BASE}/api/evaluation/train-status?method=${trainingMethod}`);
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || "Failed to fetch training status.");
        }
        setTrainingStatus((prev) => {
          const prevTerminal = prev?.status === "completed" || prev?.status === "failed";
          const nowTerminal = payload.status === "completed" || payload.status === "failed";
          if (nowTerminal && !prevTerminal) {
            queueMicrotask(() => {
              loadTrainingResults();
            });
          }
          return payload;
        });
      } catch (err) {
        if (!silent) {
          setEvaluationError(err.message || "Unexpected training-status error");
        }
      } finally {
        if (!silent) {
          setTrainingBusy(false);
        }
      }
    },
    [trainingMethod, loadTrainingResults]
  );

  const startTraining = async () => {
    setTrainingBusy(true);
    setEvaluationError("");
    try {
      const formData = new FormData();
      formData.append("method", trainingMethod);
      formData.append("profile", "best-quality");
      const response = await fetch(`${API_BASE}/api/evaluation/train`, {
        method: "POST",
        body: formData
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Failed to start training.");
      }
      setTrainingStatus(payload);
      void loadTrainingStatus({ silent: true });
    } catch (err) {
      setEvaluationError(err.message || "Unexpected training-start error");
    } finally {
      setTrainingBusy(false);
    }
  };

  const stopTraining = async () => {
    const methodToStop = trainingStatus?.method && trainingStatus?.status === "running"
      ? trainingStatus.method
      : trainingMethod;
    if (!methodToStop) {
      setEvaluationError("No active method selected to stop. Start training first.");
      return;
    }
    const confirmed = window.confirm(`Stop training for '${methodToStop}' now?`);
    if (!confirmed) {
      return;
    }
    setTrainingBusy(true);
    setEvaluationError("");
    try {
      const formData = new FormData();
      formData.append("method", methodToStop);
      const response = await fetch(`${API_BASE}/api/evaluation/train-stop`, {
        method: "POST",
        body: formData
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Failed to stop training.");
      }
      await loadTrainingStatus();
    } catch (err) {
      setEvaluationError(err.message || "Unexpected training-stop error");
    } finally {
      setTrainingBusy(false);
    }
  };

  useEffect(() => {
    if (activeView === "evaluation") {
      loadTrainingResults();
    }
  }, [activeView, loadTrainingResults]);

  useEffect(() => {
    if (activeView !== "evaluation") {
      return undefined;
    }
    void loadTrainingStatus({ silent: true });
    const id = setInterval(() => {
      void loadTrainingStatus({ silent: true });
    }, TRAINING_STATUS_POLL_MS);
    return () => clearInterval(id);
  }, [activeView, trainingMethod, loadTrainingStatus]);

  useEffect(() => {
    checkBackendHealth();
    const intervalId = setInterval(checkBackendHealth, 10000);
    return () => clearInterval(intervalId);
  }, []);

  useEffect(() => {
    let hadDisconnect = false;

    const checkFrontendHealth = async () => {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 2500);
      try {
        const response = await fetch(`${window.location.origin}/`, {
          method: "HEAD",
          cache: "no-store",
          signal: controller.signal
        });
        if (!response.ok) {
          throw new Error("Frontend ping failed");
        }
        setFrontendStatus("online");
        if (hadDisconnect) {
          window.location.reload();
          return;
        }
        hadDisconnect = false;
      } catch {
        hadDisconnect = true;
        setFrontendStatus("offline");
      } finally {
        clearTimeout(timeout);
      }
    };

    const intervalId = setInterval(checkFrontendHealth, 3000);
    return () => clearInterval(intervalId);
  }, []);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", uiTheme);
    window.localStorage.setItem("ui_theme", uiTheme);
  }, [uiTheme]);

  return (
    <div className="container">
      <h1>Tree Canopy Segmentation</h1>
      <p className="app-lead">
        {activeView === "segmentation" ? (
          <>
            <strong>Assessment 3 — Segmentation:</strong> select one of the five required CNN architectures, upload a test
            image, and compare the input with the predicted overlay and class mask (background, individual trees, tree
            groups).
          </>
        ) : (
          <>
            Use <strong>Evaluation</strong> to train models and record train / validation / test accuracy. Use{" "}
            <strong>Segmentation</strong> for qualitative inference on a single image.
          </>
        )}
      </p>
      <div className={`api-status api-status-${backendApiStatus}`}>
        Backend API: {
          backendApiStatus === "online"
            ? "Online"
            : backendApiStatus === "offline"
              ? "Offline"
              : "Checking..."
        }
      </div>
      &nbsp;
      <div className={`api-status api-status-${frontendStatus === "online" ? "online" : "checking"}`}>
        Frontend Package: {frontendStatus === "online" ? "Running" : "Reconnecting..."}
      </div>
      <div className="theme-row">
        <label htmlFor="ui-theme-select">Theme:</label>
        <select id="ui-theme-select" value={uiTheme} onChange={(e) => setUiTheme(e.target.value)}>
          <option value="blue">Blue (Default)</option>
          <option value="eco">Eco Green</option>
          <option value="dark">Dark</option>
        </select>
      </div>

      <div className="tabs">
        <button className={activeView === "evaluation" ? "tab active" : "tab"} onClick={() => setActiveView("evaluation")} type="button">
          Evaluation
        </button>
        <button className={activeView === "segmentation" ? "tab active" : "tab"} onClick={() => setActiveView("segmentation")} type="button">
          Segmentation
        </button>
      </div>

      {activeView === "segmentation" ? (
        <section className="seg-spec-panel">
          <form onSubmit={submit} className="card seg-control-card">
            <div className="seg-control-row">
              <label className="seg-field">
                <span className="seg-field-label">CNN architecture</span>
                <select value={method} onChange={(event) => setMethod(event.target.value)} aria-label="CNN architecture">
                  <option value="unet">U-Net</option>
                  <option value="deeplabv3plus">DeepLabV3</option>
                  <option value="maskrcnn">Mask R-CNN</option>
                  <option value="segformer">SegFormer</option>
                  <option value="sam2">SAM2 (Zero-shot)</option>
                </select>
              </label>
              <label className="seg-field seg-field-file">
                <span className="seg-field-label">Test image</span>
                <input type="file" accept="image/*" onChange={onFileChange} />
              </label>
              <button type="submit" disabled={loading} className="seg-submit-btn">
                {loading ? "Running…" : "Run segmentation"}
              </button>
            </div>
          </form>

          {error ? <div className="error">{error}</div> : null}

          <div className="seg-results-grid">
            <div className="card seg-result-card">
              <h2 className="seg-result-title">Input</h2>
              {previewUrl ? <img src={previewUrl} alt="Input preview" className="seg-result-img" /> : <p className="seg-placeholder">Choose an image to preview.</p>}
            </div>
            <div className="card seg-result-card">
              <h2 className="seg-result-title">Overlay</h2>
              {result?.overlay_png_base64 ? (
                <img
                  src={`data:image/png;base64,${result.overlay_png_base64}`}
                  alt="Prediction overlay"
                  className="seg-result-img"
                />
              ) : (
                <p className="seg-placeholder">Run segmentation to see the overlay.</p>
              )}
            </div>
            <div className="card seg-result-card">
              <h2 className="seg-result-title">Class mask</h2>
              {result?.mask_png_base64 ? (
                <img
                  src={`data:image/png;base64,${result.mask_png_base64}`}
                  alt="Class mask"
                  className="seg-result-img"
                />
              ) : (
                <p className="seg-placeholder">Run segmentation to see the mask.</p>
              )}
            </div>
          </div>

          <div className="card seg-legend-card">
            <h2 className="seg-legend-heading">Class legend</h2>
            {legendItems.length > 0 ? (
              <ul className="seg-legend-list">
                {legendItems.map((item) => (
                  <li className="seg-legend-item" key={item.classId}>
                    <span className="swatch" style={{ backgroundColor: item.color }} />
                    <span>{item.label}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="seg-legend-static">
                Background (black), individual trees (green), tree groups (yellow). Labels above update from the API after
                a successful run.
              </p>
            )}
          </div>

          {result ? (
            <div className="card seg-run-summary">
              <p>
                <strong>Architecture:</strong> {METHOD_LABELS[result.method] ?? result.method}
              </p>
              <p>
                <strong>Inference:</strong> {result.inference_mode}
              </p>
              {result.fallback_reason ? (
                <p className="seg-fallback-note">
                  <strong>Note:</strong> {result.fallback_reason}
                </p>
              ) : null}
            </div>
          ) : null}
        </section>
      ) : (
        <>
          <section className="card eval-spec-panel">
            <h2>Model comparison</h2>
            <p className="eval-spec-lead">
              Assessment-style summary: five architectures in fixed order, with train, validation, and test accuracy
              (or IoU when the training job stores it). Values are read from each method&apos;s{" "}
              <code>output/evaluation/&lt;method&gt;/training_results.json</code> after training completes.
            </p>

            <div className="eval-controls">
              <select
                value={trainingMethod}
                onChange={(e) => setTrainingMethod(e.target.value)}
                disabled={trainingBusy || trainingStatus?.status === "running"}
              >
                <option value="unet">U-Net</option>
                <option value="deeplabv3plus">DeepLabV3</option>
                <option value="maskrcnn">Mask R-CNN</option>
                <option value="segformer">SegFormer</option>
                <option value="sam2">SAM2 (Zero-shot)</option>
              </select>
              <span className="training-tier-label" title="Only the best-quality training preset is available.">
                Best quality
              </span>
              <button type="button" onClick={startTraining} disabled={trainingBusy || isSelectedTrainingRunning}>
                {trainingBusy || isSelectedTrainingRunning ? "Start Training (Running...)" : "Start Training"}
              </button>
              <button type="button" onClick={() => loadTrainingStatus()} disabled={trainingBusy}>
                {trainingBusy ? "Refreshing..." : "Refresh now"}
              </button>
              <button type="button" onClick={stopTraining} disabled={trainingBusy}>
                Stop Training
              </button>
              <button type="button" onClick={loadTrainingResults} disabled={trainingResultsLoading}>
                {trainingResultsLoading ? "Loading..." : "Refresh accuracy table"}
              </button>
            </div>

            <p className="profile-tip model-tip">{trainingTierNote}</p>

            <div className="spec-table-wrap">
              <table className="spec-results-table">
                <thead>
                  <tr>
                    <th scope="col">CNN Architecture</th>
                    <th scope="col">Train Accuracy</th>
                    <th scope="col">Validation Accuracy</th>
                    <th scope="col">Test Accuracy</th>
                  </tr>
                </thead>
                <tbody>
                  {EVALUATION_TABLE_ROWS.map(({ key, label }) => {
                    const r = trainingResults ? trainingResults[key] : null;
                    const fmt = (v) => {
                      if (v === null || v === undefined) return "";
                      return `${(Number(v) * 100).toFixed(2)}%`;
                    };
                    if (!r) {
                      return (
                        <tr key={key}>
                          <th scope="row">{label}</th>
                          <td colSpan={3} className="spec-empty-row">
                            Not trained yet
                          </td>
                        </tr>
                      );
                    }
                    const metricNote = r.metric_type === "iou" ? " (IoU)" : "";
                    return (
                      <tr key={key}>
                        <th scope="row">
                          <span className="spec-arch-name">{label}</span>
                          {metricNote ? <span className="metric-note spec-metric-note">{metricNote}</span> : null}
                        </th>
                        <td className={r.train_accuracy != null ? "acc-cell" : ""}>{fmt(r.train_accuracy) || "—"}</td>
                        <td className={r.val_accuracy != null ? "acc-cell" : ""}>{fmt(r.val_accuracy) || "—"}</td>
                        <td className={r.test_accuracy != null ? "acc-cell" : ""}>{fmt(r.test_accuracy) || "—"}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            <p className="eval-spec-footnote">
              The accuracy table updates when a run first reaches completed or failed. You can still use{" "}
              <strong>Refresh accuracy table</strong> anytime. SAM2 is listed as in the spec; after fine-tuning, the same
              row shows metrics from the latest run.
            </p>

            {evaluationError ? <div className="error">{evaluationError}</div> : null}

            <details className="card training-status-details">
              <summary className="training-status-summary">Training status and logs</summary>
              <p className="training-auto-refresh-hint">
                Status and log tail refresh automatically about every {TRAINING_STATUS_POLL_MS / 1000}s for the selected
                architecture while this tab is open. Use Refresh now for an immediate pull.
              </p>
              <div className="training-status-inner">
                {trainingStatus ? (
                  <>
                    <p>
                      <strong>Method:</strong> {trainingStatus.method}
                    </p>
                    <p>
                      <strong>Profile:</strong> {trainingStatus.profile ?? "—"}
                    </p>
                    <p>
                      <strong>Status:</strong> {trainingStatus.status}
                    </p>
                    <p>
                      <strong>PID:</strong> {trainingStatus.pid ?? "—"}
                    </p>
                    <p>
                      <strong>Return code:</strong> {trainingStatus.return_code ?? "—"}
                    </p>
                    <p>
                      <strong>Log:</strong> {trainingStatus.log_path}
                    </p>
                    {trainingStatus.log_tail ? <pre className="log-tail">{trainingStatus.log_tail}</pre> : null}
                  </>
                ) : (
                  <p className="training-status-empty">No status loaded yet. Start training or use Refresh Training Status.</p>
                )}
              </div>
            </details>
          </section>
        </>
      )}
    </div>
  );
}

export default App;
