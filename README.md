# Kubernetes-Based ML Pipeline Anomaly Detection

A monitoring and anomaly detection system for identifying abnormal behavior in machine learning pipelines, with a focus on concept drift and infrastructure failures.

The project combines machine learning, statistical detection methods, a streaming data workflow, and infrastructure monitoring to explore how different types of anomalies can be detected and evaluated in a Kubernetes-oriented environment.

## Overview

Machine learning systems can behave differently over time because the underlying data distribution changes or because the infrastructure supporting the application experiences problems.

This project investigates how these situations can be detected and distinguished.

The experimental setup focuses on three system states:

- Normal:the system operates under normal conditions.
- Concept Drift: changes in the data distribution may affect the behavior of the ML model.
- Infrastructure Failure: infrastructure-related problems can affect service performance and reliability.

The goal is to evaluate different anomaly detection approaches and investigate their usefulness for monitoring ML pipelines.

## Key Features

- Streaming data processing using the Elec2 dataset.
- Anomaly detection using statistical and machine learning methods.
- Comparison of Z-Score, Isolation Forest, and One-Class SVM.
- Statistical testing with the Kolmogorov–Smirnov (KS) test for distribution changes.
- FastAPI-based service for exposing application endpoints and monitoring metrics.
- Prometheus metrics collection.
- Grafana dashboards for monitoring and visualization.
- Docker-based application packaging.
- Kubernetes manifests for deploying the monitoring components and ML service.
- Evaluation scripts for benchmarking detectors and analyzing experimental results.

## Architecture

The project follows a modular architecture:

1. Data source: the Elec2 dataset provides the data used in the experiments.
2. Streaming and processing: the application processes data and maintains the state required for detection.
3. Anomaly detection:statistical and machine learning methods analyze the incoming observations.
4. API and metrics: FastAPI exposes service endpoints, while Prometheus-compatible metrics make system behavior observable.
5. Monitoring and visualization: Prometheus collects metrics and Grafana presents them through dashboards.
6. Deployment:Docker and Kubernetes configuration files support containerized execution and deployment.

### Technology Stack

| Area | Technologies |
|---|---|
| Programming language | Python |
| API | FastAPI, Uvicorn |
| Machine learning | scikit-learn |
| Data processing | NumPy, pandas, SciPy |
| Metrics | Prometheus Client, Prometheus |
| Visualization | Grafana |
| Containers | Docker, Docker Compose |
| Orchestration | Kubernetes, Minikube |
| Version control | Git, GitHub |

## Repository Structure

```text
Anomalieerkennung/
├── app/
│   ├── anomaly_detector.py
│   ├── config.py
│   ├── metrics.py
│   ├── models.py
│   ├── stream_ingestor.py
│   └── window.py
├── data/
│   └── elec2.csv
├── evaluation/
│   ├── benchmark_evaluator.py
│   ├── e2e_experiment.py
│   ├── evaluation_figures.py
│   ├── rca_evaluator.py
│   └── rca_figures.py
├── grafana/
│   ├── dashboards/
│   └── provisioning/
├── kubernetes/
│   └── k8s-manifests.yaml
├── Dockerfile
├── docker-compose.yml
├── prometheus.yml
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

Install the following tools before running the project:

- Python 3.11
- Git
- Docker Desktop
- Docker Compose

For Kubernetes deployment, you will also need:

- kubectl
- Minikube

### 1. Clone the Repository

```bash
git clone https://github.com/chischnik/Anomalieerkennung.git
cd Anomalieerkennung
```

### 2. Create a Python Virtual Environment

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

On Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the Monitoring Stack

Make sure Docker Desktop is running, then execute:

```bash
docker compose up -d
```

Check the container status:

```bash
docker compose ps
```

The application, Prometheus, and Grafana services depend on the configuration defined in `docker-compose.yml`.

**Note:** This repository contains both Docker Compose and Kubernetes configurations. Their service names, ports, and deployment settings may differ. Verify the relevant configuration before starting a particular deployment.

## Monitoring and Endpoints

When the services are running and the corresponding ports are available, the following endpoints are intended for local access:

| Service | URL | Purpose |
|---|---|---|
| Application health | http://localhost:8000/health | Check application health |
| Application state | http://localhost:8000/state | Inspect the current system state |
| Prometheus metrics | http://localhost:8000/metrics | Expose application metrics |
| Prometheus UI | http://localhost:9090 | Query collected metrics |
| Grafana | http://localhost:3000 | View monitoring dashboards |

The application endpoints depend on the actual port mappings and deployment configuration. For example, the Kubernetes setup may use `kubectl port-forward` to make services accessible locally.

## Evaluation

The repository includes scripts for evaluating anomaly detection methods and analyzing experimental behavior.

The methods considered include:

- Z-Score: identifies observations that deviate from a statistical baseline.
- Isolation Forest: detects observations that are isolated from the general data distribution.
- One-Class SVM: learns a boundary around expected observations and identifies deviations.
- Kolmogorov–Smirnov test: compares distributions to help identify potential distribution shifts.

The evaluation scripts support detector benchmarking, end-to-end experiments, and analysis of root-cause-related results.

Performance should be assessed using metrics such as precision, recall, F1-score, and detection latency. Results depend on the experimental configuration, dataset, thresholds, and scenario definitions.

## Current Scope

This repository contains the application code, monitoring configuration, deployment manifests, and evaluation scripts for the experimental project.

The current focus is on anomaly detection, observability, and experimental evaluation. Automated CI/CD deployment should be considered a separate extension unless it is explicitly implemented and tested.

## Future Improvements

Potential next steps include:

- Automated Python checks with GitHub Actions.
- Unit and integration tests for the detection pipeline.
- More systematic fault-injection experiments.
- Improved separation between model drift and infrastructure failures.
- Reproducible benchmark reports and documented evaluation results.
- A CI/CD workflow for building, testing, and deploying container images.

## Academic Context

This project was developed in the context of a Master's thesis in Computer Science and investigates anomaly detection in Kubernetes-based machine learning pipelines, with an emphasis on distinguishing concept drift from infrastructure failures.

## Author

**GitHub:** [chischnik](https://github.com/chischnik)

## License

No license has been specified yet. Unless a license is added to the repository, reuse and redistribution of this code are subject to the applicable copyright rules.
