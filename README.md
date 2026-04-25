# Machine Learning Model Deployment

End-to-end deployment of a sentiment-analysis NLP model as a production-grade prediction API on AWS, with caching, autoscaling, load testing, and observability.

<!-- markdownlint-disable MD028 -->

<p align="center">
    <!--Hugging Face-->
        <img src="https://user-images.githubusercontent.com/1393562/197941700-78283534-4e68-4429-bf94-dce7ab43a941.svg" width=7% alt="Hugging Face">
    <!--PLUS SIGN-->
        <img src="https://user-images.githubusercontent.com/1393562/190876627-da2d09cb-5ca0-4480-8eb8-830bdc0ddf64.svg" width=7% alt="Plus">
    <!--FASTAPI-->
        <img src="https://user-images.githubusercontent.com/1393562/190876570-16dff98d-ccea-4a57-86ef-a161539074d6.svg" width=7% alt="FastAPI">
    <!--PLUS SIGN-->
        <img src="https://user-images.githubusercontent.com/1393562/190876627-da2d09cb-5ca0-4480-8eb8-830bdc0ddf64.svg" width=7% alt="Plus">
    <!--REDIS LOGO-->
        <img src="https://user-images.githubusercontent.com/1393562/190876644-501591b7-809b-469f-b039-bb1a287ed36f.svg" width=7% alt="Redis">
    <!--PLUS SIGN-->
        <img src="https://user-images.githubusercontent.com/1393562/190876627-da2d09cb-5ca0-4480-8eb8-830bdc0ddf64.svg" width=7% alt="Plus">
    <!--KUBERNETES-->
        <img src="https://user-images.githubusercontent.com/1393562/190876683-9c9d4f44-b9b2-46f0-a631-308e5a079847.svg" width=7% alt="Kubernetes">
    <!--PLUS SIGN-->
        <img src="https://user-images.githubusercontent.com/1393562/190876627-da2d09cb-5ca0-4480-8eb8-830bdc0ddf64.svg" width=7% alt="Plus">
    <!--AWS-->
        <img src="https://upload.wikimedia.org/wikipedia/commons/9/93/Amazon_Web_Services_Logo.svg" width=7% alt="AWS">
    <!--PLUS SIGN-->
        <img src="https://user-images.githubusercontent.com/1393562/190876627-da2d09cb-5ca0-4480-8eb8-830bdc0ddf64.svg" width=7% alt="Plus">
    <!--k6-->
        <img src="https://user-images.githubusercontent.com/1393562/197683208-7a531396-6cf2-4703-8037-26e29935fc1a.svg" width=7% alt="K6">
    <!--PLUS SIGN-->
        <img src="https://user-images.githubusercontent.com/1393562/190876627-da2d09cb-5ca0-4480-8eb8-830bdc0ddf64.svg" width=7% alt="Plus">
    <!--GRAFANA-->
        <img src="https://user-images.githubusercontent.com/1393562/197682977-ff2ffb72-cd96-4f92-94d9-2624e29098ee.svg" width=7% alt="Grafana">
</p>

## Overview

This project packages a fine-tuned [DistilBERT](https://arxiv.org/abs/1910.01108) sentiment-classification model and serves it as a horizontally-scalable prediction API on AWS EKS. The system is designed to handle bursty traffic with low latency by combining a Redis cache, Kubernetes horizontal pod autoscaling, and an Istio-based ingress.

**Highlights**

- FastAPI service exposing typed prediction endpoints validated with Pydantic
- DistilBERT model (~300 MB) baked into the Docker image for fast cold starts
- Redis caching layer keyed per request to absorb repeated traffic
- Kubernetes deployment with init containers, health probes, and HPA
- Istio VirtualService for path-based routing
- Load tested with `k6`; metrics observed via Grafana

## Architecture

```mermaid
flowchart TB
subgraph Entire [ ]
    subgraph Userland [ ]
        User(User)
        Developer(Developer)
    end
    subgraph AWS [AWS]
        subgraph Account [Account]
            subgraph ECR [Elastic Container Registry]
                subgraph repo [Container Repo]
                    i1(image tag:4f925d7)
                    i2(image tag:29a727c):::fa
                end
            end
            subgraph k8s [Elastic Kubernetes Service]
                subgraph istio [Namespace: istio-ingress]
                    gw(gateway)
                end
                subgraph subgraph_padding1 [ ]
                    subgraph cn [Namespace: app]
                        direction TB
                        subgraph subgraph_padding2 [ ]
                            NPS3(ClusterIP: project-prediction-service):::nodes
                            subgraph ProD [Project Deployment]
                                direction TB
                                IC3(Init Container: verify-redis-dns)
                                IC4(Init Container: verify-redis-ready)
                                FA2(Project FastAPI Container):::fa

                                IC3 --> IC4 --> FA2
                            end
                            NPS1(ClusterIP: redis-service):::nodes
                            RD(Redis Deployment)

                            VS(VirtualService)

                            VS <--> NPS3

                            NPS1 <-->|Port 6379| ProD
                            NPS1 <-->|Port 6379| RD
                            NPS3 <-->|Port 8000| ProD
                        end
                    end
                end
                i1 -...- FA2
            end
        end
    end
end
gw <---> User
VS <--> gw

Developer -->|docker push| repo

classDef nodes fill:#68A063
classDef subgraph_padding fill:none,stroke:none
classDef inits fill:#cc9ef0
classDef fa fill:#00b485

style cn fill:#B6D0E2;
style RD fill:#e6584e;
style ProD fill:#FFD43B;
style k8s fill:#b77af4;
style AWS fill:#00aaff;
style Account fill:#ffbf14;
style ECR fill:#cccccc;
style repo fill:#e7e7e7;
style Userland fill:#ffffff,stroke:none;
style Entire fill:#ffffff,stroke:none;

class subgraph_padding1,subgraph_padding2 subgraph_padding
class IC3,IC4 inits
```

## Tech Stack

| Layer | Tooling |
|---|---|
| Model | DistilBERT (fine-tuned on SST-2), HuggingFace Transformers |
| API | FastAPI, Pydantic |
| Caching | Redis |
| Packaging | Poetry, Docker |
| Orchestration | Kubernetes (EKS), Kustomize, Istio |
| Cloud | AWS (ECR, EKS) |
| Load Testing | k6 |
| Observability | Grafana |

## API

**Request**

```json
{
  "text": ["example 1", "example 2"]
}
```

**Response**

```json
{
  "predictions": [
    [
      { "label": "POSITIVE", "score": 0.7127904295921326 },
      { "label": "NEGATIVE", "score": 0.2872096002101898 }
    ],
    [
      { "label": "POSITIVE", "score": 0.7186233401298523 },
      { "label": "NEGATIVE", "score": 0.2813767194747925 }
    ]
  ]
}
```

## Project Layout

```
.
├── .k8s/             # Kustomize base + overlays for Kubernetes deployment
├── mlapi/            # FastAPI application
│   ├── src/          # Application source
│   ├── tests/        # pytest suite
│   ├── trainer/      # Training reference scripts
│   ├── Dockerfile
│   └── pyproject.toml
└── load.js           # k6 load-test script
```

## Running Locally

```bash
cd mlapi
poetry install
poetry run uvicorn src.main:app --reload
```

Run the test suite:

```bash
poetry run pytest
```

## Building and Deploying

Build and push the image to ECR, then apply the Kustomize overlay:

```bash
docker build -t <ecr-repo>/project:<tag> mlapi/
docker push <ecr-repo>/project:<tag>
kubectl apply -k .k8s/overlays/<env>
```

The deployment includes:

- Init containers that verify Redis DNS and readiness before the API starts
- Liveness, readiness, and startup probes on `/project/health`
- A horizontal pod autoscaler tuned for burst traffic

## Load Testing

```bash
k6 run -e NAMESPACE=${NAMESPACE} \
  --summary-trend-stats "min,avg,med,max,p(90),p(95),p(99),p(99.99)" \
  load.js
```

## Performance Targets

Under sustained load with a warm cache, the deployed service achieves:

- ~10 requests/second sustained throughput on the predict endpoint
- p(99) latency under 2 seconds at 10 virtual users
- 95%+ cache hit rate via the Redis caching layer

## Design Notes

- **Model baked into the image.** The model is copied in at build time rather than pulled from HuggingFace at startup, keeping cold-start time low during scale-out events. In production, mounting from shared storage (EFS/S3) would be the next step.
- **Resource tuning.** Pod requests and limits are calibrated for the ~300 MB model footprint to balance scheduling headroom against memory pressure.
- **Path-based routing.** A single Istio VirtualService routes traffic to the prediction service by URL prefix, allowing additional services to be added without ingress changes.

## License

[MIT](LICENSE)
