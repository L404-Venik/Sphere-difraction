# Sphere Diffraction App — Architecture

## High-Level Structure

```
sphere-diffraction/
├── core/                              # Standalone physics library (no Qt)
│   ├── __init__.py                    # Public: calculate_S, ExperimentParameters
│   ├── parameters.py                  # Dataclasses: ExperimentParameters ...
│   ├── sphere_difraction.py           # calculate_S() — pure function
│   └── plotting_functions.py          # Pure plotting functions → matplotlib Figure
│
├── app/                               # Desktop application
│   ├── application/
│   │   ├── __init__.py
│   │   ├── controller.py              # AppController (QObject, owns state)
│   │   ├── computation.py             # ComputationManager + Worker (single QThread + queue)
│   │   └── cache.py                   # ResultCache (hash map on ExperimentParameters)
│   │
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── main_window.py             # QMainWindow (wires UI ↔ controller)
│   │   ├── parameter_panel.py         # Left panel: wavelength, layers, fidelity
│   │   ├── plot_panel.py              # Right panel: dynamic grid of FigureCanvases
│   │   ├── plot_state.py              # PlotState dataclass (appearance settings)
│   │   └── translations/
│   │       ├── sphere_diffraction_en.ts
│   │       ├── sphere_diffraction_en.qm
│   │       └── translations.qrc
│   │
│   ├── infrastructure/
│   │   ├── __init__.py
│   │   ├── serialization.py           # YAML save/load ExperimentParameters
│   │   ├── plot_exporter.py           # savefig wrappers (PNG/SVG/PDF)
│   │   ├── preferences.py             # QSettings wrapper for chart defaults
│   │   └── i18n.py                    # QTranslator setup
│   │
│   └── __main__.py                    # Entry point: QApplication → translator → controller → window
│
├── tests/
│   ├── test_core/
│   │   ├── test_coefficients.py
│   │   ├── test_difraction.py
│   │   ├── test_parameters.py
│   │   └── test_plotting.py
│   │   ├── test_special.py
│   └── test_app/
│       ├── test_controller.py
│       ├── test_computation.py
│       ├── test_cache.py
│       ├── test_serialization.py
│       └── test_ui/
│           ├── test_main_window.py
│           ├── test_parameter_panel.py
│           └── test_plot_panel.py
│
├── examples/                          # Bundled educational YAML experiments
├── benchmarks/                        # Performance benchmarks
├── docs/                              # Documentation
├── pyproject.toml
└── README.md
```

---

## Layer Rules

| Layer | Allowed imports | Forbidden |
|-------|----------------|-----------|
| `core` | `numpy`, `scipy`, `matplotlib` | `PyQt6`, `app` |
| `app/application` | `core`, `PyQt6.QtCore` | `app/ui`, `app/infrastructure` |
| `app/ui` | `core`, `PyQt6` whole | Nothing forbidden, uses controller |
| `app/infrastructure` | `core`, `PyQt6`, `tomli-w` | `app/ui`, `app/application` |

---

## Component Diagram

```
┌─────────────────────────────────────────────────────────┐
│                     UI Layer                            │
│                                                         │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐   │
│  │Param Section│  │ Plot Section │  │  Menu/ToolBar │   │
│  │             │  │              │  │               │   │
│  │ wavelength  │  │ FigureCanvas │  │ File → Save   │   │
│  │ layers[]    │  │ FigureCanvas │  │       Load    │   │
│  │             │  │ FigureCanvas │  │       Export  │   │
│  │             │  │ ... (dynamic)│  │ Settings      │   │
│  │             │  │              │  │ ...           │   │
│  └──────┬──────┘  └──────▲───────┘  └───────────────┘   │
│         │                │                              │
│    user input      display_result()                     │
│         │                │                              │
├─────────┼────────────────┼──────────────────────────────┤
│         ▼                │         Application Layer    │
│  ┌──────────────────────────────────────────────────┐   │
│  │              AppController                       │   │
│  │                                                  │   │
│  │  _current_params: ExperimentParameters           │   │
│  │  _plot_state: PlotState                          │   │
│  │  _cache: ResultCache                             │   │
│  │  _computation: ComputationManager                │   │
│  │                                                  │   │
│  │  Signals:                                        │   │
│  │    computation_started()                         │   │
│  │    computation_finished(ComputationResult) ──────┼──→ PlotPanel
│  │    computation_failed(str)                       │   │
│  │                                                  │   │
│  │  Slots:                                          │   │
│  │    set_wavelength(float)                         │   │
│  │    set_layer_thickness(int, float)               │   │
│  │    add_layer() / remove_layer(int)               │   │
│  │    set_fidelity(FidelityLevel)                   │   │
│  │    save_experiment(path) / load_experiment(path) │   │
│  │    export_plot(int, path, format)                │   │
│  └────────┬──────────────────────┬──────────────────┘   │
│           │                      │                      │
│      owns │                 owns │                      │
│           ▼                      ▼                      │
│  ┌─────────────────┐  ┌─────────────────────────────┐   │
│  │  ResultCache    │  │  ComputationManager         │   │
│  │                 │  │                             │   │
│  │  dict[params]   │  │  request_compute(params)    │   │
│  │  → result       │◄─┤  cancel_pending()           │   │
│  └────────▲────────┘  │                             │   │
│           │           │  owns QThread → Worker      │   │
│           │ cache     │  Worker.queue (latest wins) │   │
│           │ put/get   └──────────────┬──────────────┘   │
│           │                           │                 │
└───────────┼───────────────────────────┼─────────────────┘
            │                           │
            │              ┌────────────▼──────────────┐
            │              │  Worker Thread            │
            │              │  calculate_S(params)      │
            │              │  emit finished(result)    │
            │              └───────────────────────────┘
            │                           │
            │                    result │
            │                           │
┌───────────┴───────────────────────────┴──────────────────┐
│                   Domain Layer (core/)                   │
│                                                          │
│  ExperimentParameters  ──→  calculate_S()  ──→  ndarray  │
│  (frozen dataclass)          (pure function)             │
│                                                          │
│                                     │                    │
│                                     ▼                    │
│  ComputationResult  ──→  plotting_functions  ──→  Figure │
│  (ndarrays)               (pure functions)               │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## Data Flow: Slider Move to Plot Update

```
1. User drags slider / changes controll
2. QSlider.valueChanged  →  100ms debounce timer in ParameterPanel
3. Timer fires  →  AppController.set_wavelength(value)
4. Controller creates new ExperimentParameters (frozen, hashable)
5. ResultCache.get(params)?
   ├── HIT  →  emit computation_finished(cached_result)  →  jump to step 9
   └── MISS →  emit computation_started()
               ComputationManager.request_compute(params)
                          │
6. Worker thread (QThread):                           │
   queue.put(params, fidelity)                        │
   drain queue — keep only latest                     │
   result = calculate_S(params, M=fidelity.M)         │
   emit finished(ComputationResult)  ─────────────────┘
                          │
7. Controller receives finished
8. ResultCache.put(params, result)
9. Controller emits computation_finished(result)
10. PlotPanel.display_result(result):
    - For each plotting function in registry:
        fig = plot_func(...)
    - Update or create FigureCanvas per figure
    - Apply PlotState (colors, scales, grid)
    - canvas.draw_idle()
```

---

## Threading Model

```
Main Thread                          Worker Thread (QThread)
─────────────                        ──────────────────────
AppController                        Worker.run()
  │                                    │
  ├─ request_compute(params) ──►       ├─ queue.get()
  ├─ emit started()                    ├─ drain stale entries
  │                                    ├─ calculate_S(params)
  │                                    ├─ emit finished(result)
  ◄──── finished(result) ──────────────┘
  │
  ├─ cache.put(params, result)
  └─ emit computation_finished(result)

All PyQt6 widgets, matplotlib canvas: Main thread only
Pure computation: Worker thread only
Data crossing threads: ExperimentParameters (frozen) and ComputationResult (numpy arrays, created in worker, read in main)
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| `ExperimentParameters` frozen | Thread-safe handoff, hashable for cache, natural dirty-state trigger |
| Single worker thread + queue, latest-wins | 10–50ms computation, no need for thread pool; discarding stale requests keeps UI snappy |
| ResultCache by params hash | Instant undo when slider returns to previous value; free precomputation for examples |
| 100ms debounce on sliders | Reduces computation requests by 5–10× without perceptible lag |
| One `matplotlib.figure.Figure` per plot | Independent manipulation, dynamic grid layout, each exportable separately |
| Plotting functions return `Figure` | `PlotPanel` receives list of figures, adds/removes canvases dynamically — no hardcoded plot count |
| `core/` zero Qt dependency | Importable in scripts, Jupyter, alternative frontends |
| QTranslator + `self.tr()` | No third-party i18n; Qt Linguist handles everything |
| `AppController` in application layer, not UI | Testable without GUI; reusable if UI framework changes |

---

*Architecture version: 1.0 | Date: 2026-05-05*