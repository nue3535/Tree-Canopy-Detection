import { useEffect, useMemo, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

function App() {
  const [activeView, setActiveView] = useState("evaluation");
  const [file, setFile] = useState(null);
  const [method, setMethod] = useState("deeplabv3plus");
  const [previewUrl, setPreviewUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [evaluationLoading, setEvaluationLoading] = useState(false);
  const [evaluationError, setEvaluationError] = useState("");
  const [evaluationSummary, setEvaluationSummary] = useState(null);
  const [evaluationDataset, setEvaluationDataset] = useState("evaluation");
  const [evaluationMethod, setEvaluationMethod] = useState("all");
  const [trainingMethod, setTrainingMethod] = useState("deeplabv3plus");
  const [trainingProfile, setTrainingProfile] = useState("balanced");
  const [evaluationPage, setEvaluationPage] = useState(1);
  const [evaluationRowsPayload, setEvaluationRowsPayload] = useState(null);
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [trainingBusy, setTrainingBusy] = useState(false);
  const [evaluationPrecheck, setEvaluationPrecheck] = useState(null);
  const [evaluationNotice, setEvaluationNotice] = useState("");
  const [backendApiStatus, setBackendApiStatus] = useState("checking");
  const [frontendStatus, setFrontendStatus] = useState("online");

  const sortedDistribution = useMemo(() => {
    if (!result?.class_distribution) return [];
    return Object.entries(result.class_distribution).sort((a, b) => Number(a[0]) - Number(b[0]));
  }, [result]);

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

  const summaryCards = useMemo(() => {
    if (!evaluationSummary) return [];
    const methods = evaluationSummary.methods || [];
    return methods.map((m) => {
      const section = evaluationDataset === "train" ? evaluationSummary.train[m] : evaluationSummary.evaluation[m];
      return section?.summary ? { method: m, ...section.summary } : { method: m };
    });
  }, [evaluationSummary, evaluationDataset]);

  const selectedSummary = useMemo(() => {
    if (!evaluationSummary) return null;
    if (evaluationMethod === "all") return null;
    const section = evaluationDataset === "train" ? evaluationSummary.train?.[evaluationMethod] : evaluationSummary.evaluation?.[evaluationMethod];
    return section?.summary || null;
  }, [evaluationSummary, evaluationDataset, evaluationMethod]);

  const isMetricsApplicableForTable = useMemo(() => {
    if (evaluationDataset === "train") return true;
    if (!evaluationSummary) return false;
    if (evaluationMethod === "all") return Boolean(evaluationSummary.evaluation_has_ground_truth);
    return Boolean(selectedSummary?.metrics_applicable);
  }, [evaluationDataset, evaluationSummary, evaluationMethod, selectedSummary]);

  const comparisonLocked = useMemo(() => {
    return Boolean(evaluationPrecheck?.any_fallback_risk);
  }, [evaluationPrecheck]);

  const isSelectedTrainingRunning = useMemo(() => {
    return trainingStatus?.status === "running" && trainingStatus?.method === trainingMethod;
  }, [trainingStatus, trainingMethod]);

  const profileHint = useMemo(() => {
    if (trainingProfile === "fast") {
      return "Fast: shortest runtime, lowest compute, lower final accuracy.";
    }
    if (trainingProfile === "best-quality") {
      return "Best Quality: longest runtime, highest compute, strongest expected accuracy.";
    }
    return "Balanced: recommended default with good quality/time trade-off.";
  }, [trainingProfile]);

  const modelProfileHint = useMemo(() => {
    const hints = {
      deeplabv3plus: {
        fast: "DeepLabV3+ Fast: usually the quickest stable baseline.",
        balanced: "DeepLabV3+ Balanced: strong baseline for quality vs time.",
        "best-quality": "DeepLabV3+ Best Quality: longer run, best expected DeepLab accuracy."
      },
      sam2: {
        fast: "SAM2 Fast: moderate runtime; prompt-based training still compute-heavy.",
        balanced: "SAM2 Balanced: better convergence; checkpoint writes every 200 steps.",
        "best-quality": "SAM2 Best Quality: longest SAM2 run; best if you can leave it overnight."
      },
      unet: {
        fast: "U-Net Fast: very practical on CPU/GPU for quick experiments.",
        balanced: "U-Net Balanced: good default for reliable canopy segmentation.",
        "best-quality": "U-Net Best Quality: longer epochs, typically improves boundary consistency."
      },
      maskrcnn: {
        fast: "Mask R-CNN Fast: still heavy; keep expectations moderate on CPU.",
        balanced: "Mask R-CNN Balanced: robust but often the slowest among the five.",
        "best-quality": "Mask R-CNN Best Quality: highest memory/time demand; prefer GPU."
      },
      segformer: {
        fast: "SegFormer Fast: efficient transformer baseline with reasonable quality.",
        balanced: "SegFormer Balanced: strong general-purpose choice.",
        "best-quality": "SegFormer Best Quality: high compute, often top semantic quality."
      }
    };
    const byMethod = hints[trainingMethod] || hints.deeplabv3plus;
    return byMethod[trainingProfile] || byMethod.balanced;
  }, [trainingMethod, trainingProfile]);

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

  const runEvaluation = async (force = false) => {
    setEvaluationLoading(true);
    setEvaluationError("");
    setEvaluationNotice("");
    try {
      const precheckPayload = await runPrecheck();
      const response = await fetch(`${API_BASE}/api/evaluation/summary?force=${force ? "true" : "false"}`);
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Failed to run evaluation.");
      }
      setEvaluationSummary(payload);
      setEvaluationPage(1);
      await loadEvaluationPage(1, payload);
      if (precheckPayload?.any_fallback_risk) {
        setEvaluationNotice("Some methods are in fallback mode. Cross-model comparison is not fully reliable.");
      }
    } catch (err) {
      setEvaluationError(err.message || "Unexpected evaluation error");
    } finally {
      setEvaluationLoading(false);
    }
  };

  const runPrecheck = async () => {
    const response = await fetch(`${API_BASE}/api/evaluation/precheck`);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Failed to run evaluation precheck.");
    }
    setEvaluationPrecheck(payload);
    return payload;
  };

  const loadEvaluationPage = async (targetPage = evaluationPage, summaryOverride = evaluationSummary) => {
    if (!summaryOverride) return;
    setEvaluationLoading(true);
    setEvaluationError("");
    try {
      const params = new URLSearchParams({
        dataset: evaluationDataset,
        method: evaluationMethod,
        page: String(targetPage),
        page_size: "10"
      });
      const response = await fetch(`${API_BASE}/api/evaluation/page?${params.toString()}`);
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Failed to fetch evaluation page.");
      }
      setEvaluationRowsPayload(payload);
    } catch (err) {
      setEvaluationError(err.message || "Unexpected pagination error");
    } finally {
      setEvaluationLoading(false);
    }
  };

  const onDatasetChange = (value) => {
    setEvaluationDataset(value);
    setEvaluationPage(1);
  };

  const onMethodChange = (value) => {
    setEvaluationMethod(value);
    setEvaluationPage(1);
  };

  const startTraining = async () => {
    setTrainingBusy(true);
    setEvaluationError("");
    try {
      const formData = new FormData();
      formData.append("method", trainingMethod);
      formData.append("profile", trainingProfile);
      const response = await fetch(`${API_BASE}/api/evaluation/train`, {
        method: "POST",
        body: formData
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Failed to start training.");
      }
      setTrainingStatus(payload);
    } catch (err) {
      setEvaluationError(err.message || "Unexpected training-start error");
    } finally {
      setTrainingBusy(false);
    }
  };

  const loadTrainingStatus = async () => {
    setTrainingBusy(true);
    setEvaluationError("");
    try {
      const response = await fetch(`${API_BASE}/api/evaluation/train-status?method=${trainingMethod}`);
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Failed to fetch training status.");
      }
      setTrainingStatus(payload);
    } catch (err) {
      setEvaluationError(err.message || "Unexpected training-status error");
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
    if (activeView !== "evaluation" || !evaluationSummary) return;
    loadEvaluationPage();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeView, evaluationDataset, evaluationMethod, evaluationPage]);

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

  return (
    <div className="container">
      <h1>Tree Canopy Segmentation</h1>
      <p>Run segmentation and evaluate DeepLabV3+, SAM2, U-Net, Mask R-CNN, and SegFormer.</p>
      <div className={`api-status api-status-${backendApiStatus}`}>
        Backend API: {
          backendApiStatus === "online"
            ? "Online"
            : backendApiStatus === "offline"
              ? "Offline"
              : "Checking..."
        }
      </div>
      <div className={`api-status api-status-${frontendStatus === "online" ? "online" : "checking"}`}>
        Frontend Package: {frontendStatus === "online" ? "Running" : "Reconnecting..."}
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
        <>
          <form onSubmit={submit} className="card">
            <select value={method} onChange={(event) => setMethod(event.target.value)}>
              <option value="deeplabv3plus">DeepLabV3+</option>
              <option value="sam2">SAM2</option>
              <option value="unet">U-Net</option>
              <option value="maskrcnn">Mask R-CNN</option>
              <option value="segformer">SegFormer</option>
            </select>
            <input type="file" accept="image/*" onChange={onFileChange} />
            <button type="submit" disabled={loading}>
              {loading ? "Running..." : "Segment Image"}
            </button>
          </form>

          {error ? <div className="error">{error}</div> : null}

          <div className="grid">
            <div className="card">
              <h2>Input</h2>
              {previewUrl ? <img src={previewUrl} alt="Input preview" /> : <p>No image selected.</p>}
            </div>

            <div className="card">
              <h2>Segmentation Overlay</h2>
              {result?.overlay_png_base64 ? (
                <img src={`data:image/png;base64,${result.overlay_png_base64}`} alt="Overlay result" />
              ) : (
                <p>No prediction yet.</p>
              )}
            </div>

            <div className="card">
              <h2>Segmentation Mask</h2>
              {result?.mask_png_base64 ? (
                <>
                  <img src={`data:image/png;base64,${result.mask_png_base64}`} alt="Segmentation mask result" />
                  <div className="legend">
                    {legendItems.map((item) => (
                      <div className="legend-item" key={item.classId}>
                        <span className="swatch" style={{ backgroundColor: item.color }} />
                        <span>{item.label}</span>
                      </div>
                    ))}
                  </div>
                </>
              ) : (
                <p>No prediction yet.</p>
              )}
            </div>

            <div className="card">
              <h2>Metadata</h2>
              {result ? (
                <>
                  <p>
                    <strong>Method:</strong> {result.method}
                  </p>
                  <p>
                    <strong>Inference mode:</strong> {result.inference_mode}
                  </p>
                  {result.fallback_reason ? (
                    <p>
                      <strong>Note:</strong> {result.fallback_reason}
                    </p>
                  ) : null}
                  <p>
                    <strong>Scene:</strong> {result.scene_label} (class {result.scene_class})
                  </p>
                  <h3>Class Distribution</h3>
                  <ul>
                    {sortedDistribution.map(([classId, ratio]) => (
                      <li key={classId}>
                        Class {classId}: {(ratio * 100).toFixed(2)}%
                      </li>
                    ))}
                  </ul>
                </>
              ) : (
                <p>No metadata yet.</p>
              )}
            </div>
          </div>
        </>
      ) : (
        <>
          <div className="card eval-controls">
            <select
              value={trainingMethod}
              onChange={(e) => setTrainingMethod(e.target.value)}
              disabled={trainingBusy || evaluationLoading || trainingStatus?.status === "running"}
            >
              <option value="deeplabv3plus">DeepLabV3+</option>
              <option value="sam2">SAM2</option>
              <option value="unet">U-Net</option>
              <option value="maskrcnn">Mask R-CNN</option>
              <option value="segformer">SegFormer</option>
            </select>
            <select
              value={trainingProfile}
              onChange={(e) => setTrainingProfile(e.target.value)}
              disabled={trainingBusy || evaluationLoading || trainingStatus?.status === "running"}
            >
              <option value="fast">Fast</option>
              <option value="balanced">Balanced</option>
              <option value="best-quality">Best Quality</option>
            </select>
            <span
              className="profile-help"
              title={
                "Fast: least time/compute. Balanced: recommended default. Best Quality: most time/compute with strongest expected accuracy."
              }
            >
              Profile help
            </span>
            <button type="button" onClick={startTraining} disabled={trainingBusy || evaluationLoading || isSelectedTrainingRunning}>
              {trainingBusy || isSelectedTrainingRunning ? "Start Training (Running...)" : "Start Training"}
            </button>
            <button type="button" onClick={loadTrainingStatus} disabled={trainingBusy || evaluationLoading}>
              {trainingBusy ? "Refreshing..." : "Refresh Training Status"}
            </button>
            <button type="button" onClick={stopTraining} disabled={evaluationLoading}>
              Stop Training
            </button>
            <button type="button" onClick={() => runEvaluation(true)} disabled={evaluationLoading}>
              {evaluationLoading ? "Running Evaluation..." : "Run / Refresh Full Evaluation"}
            </button>
            {/* Dataset selector hidden for fixed train->evaluation workflow.
                Keep this block commented for easy future re-enable. */}
            {/*
            <select value={evaluationDataset} onChange={(e) => onDatasetChange(e.target.value)}>
              <option value="train">Training Data</option>
              <option value="evaluation">Evaluation Data</option>
            </select>
            */}
            {/* <select value={evaluationMethod} onChange={(e) => onMethodChange(e.target.value)}>
              <option value="all">All Methods</option>
              <option value="deeplabv3plus">DeepLabV3+</option>
              <option value="sam2">SAM2</option>
              <option value="unet">U-Net</option>
              <option value="maskrcnn">Mask R-CNN</option>
              <option value="segformer">SegFormer</option>
            </select> */}
            {/* <button type="button" onClick={loadEvaluationPage} disabled={!evaluationSummary || evaluationLoading}>
              Load Page
            </button> */}
          </div>
          <div className="profile-tip">{profileHint}</div>
          <div className="profile-tip model-tip">{modelProfileHint}</div>

          {evaluationNotice ? <div className="error">{evaluationNotice}</div> : null}

          {evaluationPrecheck ? (
            <div className="card">
              <h2>Model Precheck</h2>
              {Object.entries(evaluationPrecheck.methods).map(([name, status]) => (
                <p key={`precheck-${name}`}>
                  <strong>{name}:</strong>{" "}
                  {status.ready_for_model_inference ? "Trained checkpoint ready" : "Fallback risk"}
                  {status.checkpoint_path ? ` | checkpoint: ${status.checkpoint_path}` : ""}
                </p>
              ))}
              {comparisonLocked ? (
                <p><strong>Comparison:</strong> Locked for strict model comparison because at least one method is fallback.</p>
              ) : (
                <p><strong>Comparison:</strong> Ready for strict cross-model comparison.</p>
              )}
            </div>
          ) : null}

          {/* <div className="card eval-controls">
            <button type="button" onClick={trainAllAndEvaluate} disabled={trainingBusy || evaluationLoading}>
              {trainingBusy ? "Running All..." : "Train All + Evaluate"}
            </button>
            <button type="button" onClick={startTraining} disabled={trainingBusy}>
              {trainingBusy ? "Starting..." : `Start Training (${evaluationMethod})`}
            </button>
            <button type="button" onClick={loadTrainingStatus} disabled={trainingBusy}>
              {trainingBusy ? "Checking..." : `Check Training Status (${evaluationMethod})`}
            </button>
            <button type="button" onClick={stopTraining} disabled={trainingBusy}>
              {trainingBusy ? "Stopping..." : `Stop Training (${evaluationMethod})`}
            </button>
          </div> */}

          {trainingStatus ? (
            <div className="card">
              <h2>Training Status</h2>
              <p><strong>Method:</strong> {trainingStatus.method}</p>
              <p><strong>Profile:</strong> {trainingStatus.profile ?? "-"}</p>
              <p><strong>Status:</strong> {trainingStatus.status}</p>
              <p><strong>PID:</strong> {trainingStatus.pid ?? "-"}</p>
              <p><strong>Return Code:</strong> {trainingStatus.return_code ?? "-"}</p>
              <p><strong>Log:</strong> {trainingStatus.log_path}</p>
              {trainingStatus.log_tail ? <pre className="log-tail">{trainingStatus.log_tail}</pre> : null}
            </div>
          ) : null}

          {evaluationSummary ? (
            <div className="card">
              <p>
                <strong>Train annotations:</strong> {evaluationSummary.train_annotations_path}
              </p>
              <p>
                <strong>Evaluation annotations:</strong> {evaluationSummary.evaluation_annotations_path}
              </p>
              <p>
                <strong>Evaluation GT available:</strong> {evaluationSummary.evaluation_has_ground_truth ? "Yes" : "No"}
              </p>
            </div>
          ) : null}

          {evaluationError ? <div className="error">{evaluationError}</div> : null}

          <div className="grid">
            {summaryCards.map((card) => (
              <div className="card" key={`${evaluationDataset}-${card.method}`}>
                <h2>{card.method}</h2>
                <p><strong>Images:</strong> {card.num_images ?? "-"}</p>
                {evaluationDataset === "train" ? (
                  <>
                    <p><strong>Accuracy:</strong> {card.accuracy ?? "-"}</p>
                    <p><strong>Macro F1:</strong> {card.macro_f1 ?? "-"}</p>
                    <p><strong>Macro IoU:</strong> {card.macro_iou ?? "-"}</p>
                  </>
                ) : (
                  <>
                    {card.metrics_applicable ? (
                      <>
                        <p><strong>Accuracy:</strong> {card.accuracy ?? "-"}</p>
                        <p><strong>Macro F1:</strong> {card.macro_f1 ?? "-"}</p>
                        <p><strong>Macro IoU:</strong> {card.macro_iou ?? "-"}</p>
                      </>
                    ) : (
                      <>
                        <p><strong>Avg Tree Ratio:</strong> {card.avg_tree_ratio ?? "-"}</p>
                        <p><strong>Avg Group Ratio:</strong> {card.avg_group_ratio ?? "-"}</p>
                      </>
                    )}
                  </>
                )}
                {card.note ? <p><strong>Note:</strong> {card.note}</p> : null}
                <p><strong>Avg Inference (ms):</strong> {card.avg_inference_ms ?? "-"}</p>
                <p><strong>Inference Modes:</strong> {JSON.stringify(card.inference_mode_counts || {})}</p>
                <p><strong>Fallback Images:</strong> {card.fallback_image_count ?? "-"}</p>
              </div>
            ))}
          </div>

          {(evaluationDataset === "train" || selectedSummary?.metrics_applicable) && evaluationSummary ? (
            <div className="card">
              <h2>Confusion Matrix (Selected Method)</h2>
              {selectedSummary?.confusion_matrix ? (
                <table className="matrix-table">
                  <thead>
                    <tr>
                      <th>GT \ Pred</th>
                      <th>Background</th>
                      <th>Tree</th>
                      <th>Tree Group</th>
                    </tr>
                  </thead>
                  <tbody>
                    {selectedSummary.confusion_matrix.map((row, idx) => (
                      <tr key={`cm-${idx}`}>
                        <td>{idx === 0 ? "Background" : idx === 1 ? "Tree" : "Tree Group"}</td>
                        {row.map((v, j) => (
                          <td key={`cm-${idx}-${j}`}>{v}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <p>Confusion matrix is not applicable for unlabeled evaluation images.</p>
              )}
            </div>
          ) : null}

          <div className="card">
            <h2>Evaluation Outcomes</h2>
            {!evaluationRowsPayload ? (
              <p>Run evaluation and load a page to view per-image outcomes.</p>
            ) : (
              <>
                <table className="matrix-table">
                  <thead>
                    <tr>
                      <th>File</th>
                      <th>Method</th>
                      <th>Inference</th>
                      {isMetricsApplicableForTable ? <th>Pixel Accuracy</th> : <th>Tree Ratio</th>}
                    </tr>
                  </thead>
                  <tbody>
                    {evaluationRowsPayload.items.map((row) => (
                      <tr key={`${row.file_name}-${row.method}`}>
                        <td>{row.file_name}</td>
                        <td>{row.method}</td>
                        <td>{row.inference_mode}</td>
                        {isMetricsApplicableForTable ? (
                          <td>{row.pixel_accuracy ?? "-"}</td>
                        ) : (
                          <td>{row.tree_ratio ?? "-"}</td>
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div className="pager">
                  <button
                    type="button"
                    disabled={evaluationRowsPayload.page <= 1 || evaluationLoading}
                    onClick={() => setEvaluationPage((p) => Math.max(1, p - 1))}
                  >
                    Prev
                  </button>
                  <span>
                    Page {evaluationRowsPayload.page} / {evaluationRowsPayload.total_pages}
                  </span>
                  <button
                    type="button"
                    disabled={evaluationRowsPayload.page >= evaluationRowsPayload.total_pages || evaluationLoading}
                    onClick={() =>
                      setEvaluationPage((p) =>
                        Math.min(evaluationRowsPayload.total_pages, p + 1)
                      )
                    }
                  >
                    Next
                  </button>
                </div>
              </>
            )}
          </div>
        </>
      )}
    </div>
  );
}

export default App;
