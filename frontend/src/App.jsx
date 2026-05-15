import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as UTIF from "utif";

const TRAINING_STATUS_POLL_MS = 2500;
const TRAINING_STATUS_POLL_RUNNING_MS = 800;

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

/** Rows for the evaluation accuracy table; order is user-facing priority. */
const EVALUATION_TABLE_ROWS = [
  { key: "maskrcnn", label: "Mask R-CNN" },
  { key: "sam2", label: "SAM2" }
];

const METHOD_LABELS = Object.fromEntries(EVALUATION_TABLE_ROWS.map(({ key, label }) => [key, label]));
const SEGMENT_SENSITIVITY_OPTIONS = [
  { key: "conservative", label: "Conservative" },
  { key: "balanced", label: "Balanced" },
  { key: "recall", label: "Recall" }
];

/** Matches API `class_colors` / mask visualization (background, tree, tree group). */
const STATIC_CLASS_LEGEND = [
  { classId: "0", label: "Background", color: "#000000" },
  { classId: "1", label: "Tree", color: "#00ff00" },
  { classId: "2", label: "Tree Group", color: "#ffff00" }
];

const TIFF_EXTENSIONS = [".tif", ".tiff"];

function isTiffFile(file) {
  const name = (file?.name || "").toLowerCase();
  const type = (file?.type || "").toLowerCase();
  return TIFF_EXTENSIONS.some((ext) => name.endsWith(ext)) || type.includes("tif") || type.includes("tiff");
}

async function tiffFileToPngDataUrl(file) {
  const buffer = await file.arrayBuffer();
  const ifds = UTIF.decode(buffer);
  if (!ifds.length) {
    throw new Error("No TIFF image frames found.");
  }
  UTIF.decodeImage(buffer, ifds[0]);
  const width = Number(ifds[0].width || 0);
  const height = Number(ifds[0].height || 0);
  if (width <= 0 || height <= 0) {
    throw new Error("Invalid TIFF dimensions.");
  }
  const rgba = UTIF.toRGBA8(ifds[0]);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("Could not create canvas for TIFF preview.");
  }
  const imageData = new ImageData(new Uint8ClampedArray(rgba), width, height);
  ctx.putImageData(imageData, 0, 0);
  return canvas.toDataURL("image/png");
}

function App() {
  const getInitialTheme = () => {
    if (typeof window === "undefined") return "blue";
    const saved = window.localStorage.getItem("ui_theme");
    return saved === "blue" || saved === "eco" || saved === "dark" ? saved : "blue";
  };

  const [activeView, setActiveView] = useState("evaluation");
  const [file, setFile] = useState(null);
  const [method, setMethod] = useState("sam2");
  const [segmentSensitivity, setSegmentSensitivity] = useState("balanced");
  const [previewUrl, setPreviewUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [evaluationError, setEvaluationError] = useState("");
  const [trainingMethod, setTrainingMethod] = useState("sam2");
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [trainingBusy, setTrainingBusy] = useState(false);
  const [backendApiStatus, setBackendApiStatus] = useState("checking");
  const [frontendStatus, setFrontendStatus] = useState("online");
  const [uiTheme, setUiTheme] = useState(getInitialTheme);
  const [trainingResults, setTrainingResults] = useState(null);
  const [trainingResultsLoading, setTrainingResultsLoading] = useState(false);
  const [segmentPrecheck, setSegmentPrecheck] = useState(null);
  const [trainingLogExpanded, setTrainingLogExpanded] = useState(false);
  const logTailRef = useRef(null);
  const previewObjectUrlRef = useRef("");

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

  const legendDisplayItems = legendItems.length > 0 ? legendItems : STATIC_CLASS_LEGEND;

  const loadSegmentPrecheck = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/evaluation/precheck`);
      const payload = await response.json();
      if (response.ok && payload?.methods) {
        setSegmentPrecheck(payload.methods);
      } else {
        setSegmentPrecheck(null);
      }
    } catch {
      setSegmentPrecheck(null);
    }
  }, []);

  const isSelectedTrainingRunning = useMemo(() => {
    return trainingStatus?.status === "running" && trainingStatus?.method === trainingMethod;
  }, [trainingStatus, trainingMethod]);

  const trainingPollMs = useMemo(() => {
    if (activeView !== "evaluation") {
      return TRAINING_STATUS_POLL_MS;
    }
    return trainingStatus?.status === "running" ? TRAINING_STATUS_POLL_RUNNING_MS : TRAINING_STATUS_POLL_MS;
  }, [activeView, trainingStatus?.status]);

  const trainingTierNote = useMemo(() => {
    if (trainingMethod === "maskrcnn") {
      return (
        "Mask R-CNN learns detection + instance segmentation together: RPN, box regression, " +
        "classification (individual_tree vs group_of_trees), and masks. Default 500 epochs with early stopping " +
        "on val loss (patience 50). Logs print torchvision loss components and a summary line " +
        "(Epoch NN/500 | train_loss | val_loss | Patience). On Windows, DataLoader workers default to 0 to avoid " +
        "subprocess MemoryError. Checkpoints: checkpoints_mask_rcnn/best_model.pth (saved on best val_loss)."
      );
    }
    return (
      "SAM2 fine-tunes with GT box prompts (epoch loop, default LR 1e-5, early stopping on val loss with default patience 50; " +
      "set SAM2_EARLY_STOPPING_PATIENCE=0 to disable). " +
      "The “Best quality” label is only the selected profile name—it does not switch SAM2 hyperparameters. " +
      "On Start Training, the API copies the resolved YAML into checkpoints_sam2 and downloads the matching Meta base .pt " +
      "if it is missing (requires network the first time). Expect long runs and high VRAM."
    );
  }, [trainingMethod]);

  const onFileChange = async (event) => {
    const selected = event.target.files?.[0] || null;
    setFile(selected);
    setResult(null);
    setError("");
    if (previewObjectUrlRef.current) {
      URL.revokeObjectURL(previewObjectUrlRef.current);
      previewObjectUrlRef.current = "";
    }
    if (selected) {
      try {
        if (isTiffFile(selected)) {
          const dataUrl = await tiffFileToPngDataUrl(selected);
          setPreviewUrl(dataUrl);
        } else {
          const objectUrl = URL.createObjectURL(selected);
          previewObjectUrlRef.current = objectUrl;
          setPreviewUrl(objectUrl);
        }
      } catch {
        const objectUrl = URL.createObjectURL(selected);
        previewObjectUrlRef.current = objectUrl;
        setPreviewUrl(objectUrl);
        setError("Could not render local TIFF preview in browser, but segmentation upload still works.");
      }
    } else {
      setPreviewUrl("");
    }
  };

  useEffect(() => {
    return () => {
      if (previewObjectUrlRef.current) {
        URL.revokeObjectURL(previewObjectUrlRef.current);
        previewObjectUrlRef.current = "";
      }
    };
  }, []);

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
      formData.append("segment_sensitivity", segmentSensitivity);
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
      setTrainingLogExpanded(true);
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
    }, trainingPollMs);
    return () => clearInterval(id);
  }, [activeView, trainingMethod, loadTrainingStatus, trainingPollMs]);

  useEffect(() => {
    const el = logTailRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [trainingStatus?.log_tail]);

  useEffect(() => {
    if (activeView === "evaluation" && trainingStatus?.status === "running") {
      setTrainingLogExpanded(true);
    }
  }, [activeView, trainingStatus?.status]);

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

  useEffect(() => {
    if (activeView === "segmentation" && backendApiStatus === "online") {
      void loadSegmentPrecheck();
    }
  }, [activeView, backendApiStatus, loadSegmentPrecheck]);

  useEffect(() => {
    setResult(null);
    setError("");
  }, [method]);

  return (
    <div className="container">
      <h1>Tree Canopy Segmentation</h1>
      <p className="app-lead">
        {activeView === "segmentation" ? (
          <>
            <strong>SAM2 segmentation</strong> uses the same fine-tuned weights produced on the Evaluation page
            (<code>checkpoints_sam2/sam2_finetuned_tree_canopy.pt</code>, or best/latest training snapshots if the export
            is not present yet). Upload your own image to run inference; results update automatically after training
            without restarting the server when possible.
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
      {/* <div className="theme-row">
        <label htmlFor="ui-theme-select">Theme:</label>
        <select id="ui-theme-select" value={uiTheme} onChange={(e) => setUiTheme(e.target.value)}>
          <option value="blue">Blue (Default)</option>
          <option value="eco">Eco Green</option>
          <option value="dark">Dark</option>
        </select>
      </div> */}

      <div className="tabs" role="tablist" aria-label="Main sections">
        <button
          className={activeView === "evaluation" ? "tab active" : "tab"}
          onClick={() => setActiveView("evaluation")}
          type="button"
          role="tab"
          aria-selected={activeView === "evaluation"}
        >
          Evaluation
        </button>
        <button
          className={activeView === "segmentation" ? "tab active" : "tab"}
          onClick={() => setActiveView("segmentation")}
          type="button"
          role="tab"
          aria-selected={activeView === "segmentation"}
        >
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
                  <option value="maskrcnn">Mask R-CNN (detect + segment)</option>
                  <option value="sam2">SAM2 (prompted segmentation)</option>
                </select>
              </label>
              <label className="seg-field">
                <span className="seg-field-label">Sensitivity</span>
                <select
                  value={segmentSensitivity}
                  onChange={(event) => setSegmentSensitivity(event.target.value)}
                  aria-label="Segmentation sensitivity"
                >
                  {SEGMENT_SENSITIVITY_OPTIONS.map((opt) => (
                    <option key={opt.key} value={opt.key}>
                      {opt.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="seg-field seg-field-file">
                <span className="seg-field-label">Test image</span>
                <input type="file" accept="image/*,.tif,.tiff" onChange={onFileChange} />
              </label>
              <button type="submit" disabled={loading} className="seg-submit-btn">
                {loading ? "Running…" : "Run segmentation"}
              </button>
            </div>
            {segmentPrecheck?.[method] != null ? (
              <p
                className={
                  segmentPrecheck[method].ready_for_model_inference ? "seg-precheck seg-precheck-ok" : "seg-precheck seg-precheck-warn"
                }
              >
                {segmentPrecheck[method].ready_for_model_inference
                  ? `Weights ready: ${segmentPrecheck[method].checkpoint_path || "configured"}.`
                  : `Model may use fallback: ${segmentPrecheck[method].reason || "checkpoint or dependency missing."}`}
              </p>
            ) : null}
          </form>

          {error ? <div className="error">{error}</div> : null}

          <div className="seg-results-grid">
            <div className="card seg-result-card">
              <h2 className="seg-result-title">Input</h2>
              {previewUrl ? <img src={previewUrl} alt="Input preview" className="seg-result-img" /> : <p className="seg-placeholder">Choose an image to preview.</p>}
              <p className="seg-legend-note">TIFF preview decoded locally in your browser.</p>
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
            <ul className="seg-legend-list">
              {legendDisplayItems.map((item) => (
                <li className="seg-legend-item" key={item.classId}>
                  <span className="swatch" style={{ backgroundColor: item.color }} />
                  <span>{item.label}</span>
                </li>
              ))}
            </ul>
            {result?.class_distribution ? (
              <p className="seg-legend-note">
                Pixel fractions after last run — background: {(result.class_distribution["0"] * 100).toFixed(1)}%, trees:{" "}
                {(result.class_distribution["1"] * 100).toFixed(1)}%, groups: {(result.class_distribution["2"] * 100).toFixed(1)}%.
              </p>
            ) : (
              <p className="seg-legend-note">Overlay and class mask use the same colors as this legend.</p>
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
            <h2>Model Training & Evaluation</h2>
            <p className="eval-spec-lead">
              <strong>Mask R-CNN</strong> trains <strong>object detection + instance segmentation</strong> (boxes, two
              tree classes, and masks). <strong>SAM2</strong> here is <strong>segmentation with GT box prompts</strong>{" "}
              (no learned localizer). The table shows train / validation / test metrics from each method’s{" "}
              <code>training_results.json</code> (SAM2: mean IoU from prompted eval; Mask R-CNN: mask accuracy-style
              summary from the workflow). Reload with <strong>Refresh accuracy table</strong> or when the UI detects a
              completed or failed run.
            </p>

            <div className="eval-controls">
              <select
                value={trainingMethod}
                onChange={(e) => setTrainingMethod(e.target.value)}
                disabled={trainingBusy || trainingStatus?.status === "running"}
              >
                <option value="maskrcnn">Mask R-CNN</option>
                <option value="sam2">SAM2</option>
              </select>
              <span className="training-tier-label" title="Only the best-quality training preset is available.">
                Best quality
              </span>
              <button type="button" onClick={startTraining} disabled={trainingBusy || isSelectedTrainingRunning}>
                {trainingBusy || isSelectedTrainingRunning ? "Start Training (Running...)" : "Start Training"}
              </button>
              <button type="button" onClick={stopTraining} disabled={trainingBusy}>
                Stop Training
              </button>
            </div>

            <p className="profile-tip model-tip">{trainingTierNote}</p>

            <div className="accuracy-table-toolbar">
              <button type="button" onClick={loadTrainingResults} disabled={trainingResultsLoading}>
                {trainingResultsLoading ? "Loading..." : "Refresh accuracy table"}
              </button>
            </div>

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
                    const metricNote =
                      r.metric_type === "mean_iou" ? " (mean IoU)" : r.metric_type === "iou" ? " (IoU)" : "";
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
              The accuracy table updates when a run first reaches completed or failed. Use{" "}
              <strong>Refresh accuracy table</strong> anytime to reload metrics from the latest training run.
            </p>

            {evaluationError ? <div className="error">{evaluationError}</div> : null}

            <details
              className="card training-status-details"
              open={trainingLogExpanded}
              onToggle={(e) => setTrainingLogExpanded(e.currentTarget.open)}
            >
              <summary className="training-status-summary">Training status and logs</summary>
              <p className="training-auto-refresh-hint">
                Status and log tail refresh automatically about every{" "}
                {trainingStatus?.status === "running"
                  ? `${TRAINING_STATUS_POLL_RUNNING_MS / 1000}s while training is running`
                  : `${TRAINING_STATUS_POLL_MS / 1000}s`}{" "}
                on the Evaluation tab. After you start training, this panel opens so logs update in near real time.
              </p>
              <div className="training-status-actions">
                <button type="button" onClick={() => loadTrainingStatus()} disabled={trainingBusy}>
                  {trainingBusy ? "Refreshing..." : "Refresh now"}
                </button>
              </div>
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
                    {trainingStatus.log_tail ? (
                      <pre ref={logTailRef} className="log-tail">
                        {trainingStatus.log_tail}
                      </pre>
                    ) : null}
                  </>
                ) : (
                  <p className="training-status-empty">No status loaded yet. Start training or press Refresh now above.</p>
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
