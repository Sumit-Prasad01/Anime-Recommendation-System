# Anime Recommendation System

A deep learning--based anime recommendation platform with a user-facing
web app, experiment tracking, and full CI/CD deployment on Kubernetes.

This repository includes:\
- A TensorFlow neural recommender engine\
- A Flask frontend + HTML + Tailwind CSS UI\
- Experiment tracking via Comet.ml\
- Automation & deployment using Jenkins, Docker, Google Cloud, and
Google Kubernetes Engine

------------------------------------------------------------------------

## Table of Contents

1.  [Features](#features)\
2.  [Architecture & Components](#architecture--components)\
3.  [Getting Started](#getting-started)
    -   [Prerequisites](#prerequisites)\
    -   [Installation & Setup](#installation--setup)\
    -   [Running Locally](#running-locally)\
4.  [Model & Training](#model--training)
    -   [Data](#data)\
    -   [Model Architecture](#model-architecture)\
    -   [Experiment Tracking](#experiment-tracking)\
5.  [Web App](#web-app)
    -   [Flask + Routes](#flask--routes)\
    -   [Frontend / Templates /
        Tailwind](#frontend--templates--tailwind)\
6.  [Deployment & DevOps](#deployment--devops)
    -   [Docker & Containerization](#docker--containerization)\
    -   [CI/CD with Jenkins](#cicd-with-jenkins)\
    -   [Deployment to GKE (Google Kubernetes
        Engine)](#deployment-to-gke-google-kubernetes-engine)\
    -   [Infrastructure / Pipeline
        Files](#infrastructure--pipeline-files)\
7.  [Usage / Endpoints](#usage--endpoints)\
8.  [Configuration & Environment
    Variables](#configuration--environment-variables)\
9.  [Monitoring & Logging](#monitoring--logging)\
10. [Acknowledgements]

------------------------------------------------------------------------

## Project Structure
```bash
├── artifacts/             # Model artifacts, logs, output from training
├── config/                # Configuration files (e.g. for GCP, credentials, hyperparameters)
├── custom_jenkins/        # Jenkins pipeline definitions or scripts
├── notebooks/             # Exploratory data analysis, experimentation
├── pipeline/              # Pipeline code (data ingestion, transformation, training, etc.)
├── src/                   # Source code modules (data, models, utils)
├── templates/             # Flask API templates or HTML templates
├── utils/                 # Utility scripts and helpers
├── Dockerfile             # For creating container to serve model
├── Jenkinsfile            # CI/CD pipeline script
├── application.py         # Entry point for serving predictions (Flask app)
├── requirements.txt       # Python package dependencies
└── setup.py               # Package installation script
```


## Features

-   Personalized anime recommendations using deep learning (TensorFlow)\
-   Ability for users to view and rate anime\
-   Web interface built with Flask + HTML + Tailwind CSS\
-   Experiment tracking with Comet.ml (tracking metrics,
    hyperparameters, model artifacts)\
-   Dockerized application and model server\
-   CI/CD pipeline using Jenkins\
-   Deployment to Google Cloud Platform (GCP) + Kubernetes\
-   (Optional) Scalability and fault tolerance via Kubernetes

------------------------------------------------------------------------

## Architecture & Components

Here is a high-level overview of how the system is structured:

    +----------------------+       +----------------------+
    |  User / Frontend UI  | <-->  |     Flask Backend     |
    +----------------------+       +----------------------+
                                           |
                                           v
                                +---------------------------+
                                |  Recommendation Engine     |
                                |  (TensorFlow / Python)     |
                                +---------------------------+
                                           |
                                           v
                                +---------------------------+
                                |  Experiment Tracking (Comet) |
                                +---------------------------+
      
      +-----------------------+       +-------------------------+
      | Docker / Container     | <---> | Jenkins & CI/CD pipeline |
      +-----------------------+       +-------------------------+
                                           |
                                           v
                                +---------------------------+
                                |  Google Cloud / Kubernetes |
                                +---------------------------+

------------------------------------------------------------------------

## Getting Started

### Prerequisites

-   Python 3.8+\
-   `pip` / `venv`\
-   Docker\
-   (Optional, for local Kubernetes) `kubectl`, `minikube`\
-   A Google Cloud project with billing enabled, GKE API enabled\
-   A Jenkins server (or local Jenkins setup)\
-   A Comet.ml account / API key

### Installation & Setup

``` bash
git clone https://github.com/Sumit-Prasad01/Anime-Recommendation-System.git
cd Anime-Recommendation-System
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set up Comet configuration in `.env`:
```
    COMET_API_KEY=your_comet_api_key
    COMET_PROJECT_NAME=anime-recommender
    COMET_WORKSPACE=your_workspace
```

### Running Locally

``` bash
python application.py
```

Visit `http://localhost:5000`.

------------------------------------------------------------------------

## Model & Training

-   Dataset: Anime dataset from MyAnimeList / Kaggle\
-   TensorFlow-based neural recommender (embeddings, dense layers,
    dot-product scoring)\
-   Loss: MSE / ranking loss\
-   Hyperparameters tracked via Comet.ml

``` python
from comet_ml import Experiment
experiment = Experiment(api_key=..., project_name=..., workspace=...)
experiment.log_parameters(params)
experiment.log_metrics({"loss": loss_value})
```

------------------------------------------------------------------------

## Deployment & DevOps

-   **Docker:** Containerizes the app and model server\
-   **Jenkins:** Automates CI/CD pipeline (build, test, push, deploy)\
-   **GKE:** Hosts containers on Google Kubernetes Engine\
-   **Pipeline:** Includes YAML deployment manifests and Jenkinsfile

------------------------------------------------------------------------

## Usage / Endpoints

    POST '/'
    Input: { "user_id": 123, "top_k": 10 }
    Response: [ {"title": "Naruto"}, ... ]

------------------------------------------------------------------------

## Acknowledgements

-   TensorFlow, Flask, TailwindCSS\
-   Comet.ml for experiment tracking\
-   Jenkins, Docker, Google Cloud for deployment



