# Modular SOTA Analysis: Human Motion Capture & Biomechanics (2026)

## Executive Summary

The state of the art in 2026 is **modular, not monolithic**. No single mature system
combines monocular RGB, LiDAR, biomechanical body models, temporal generative priors,
and RL-based physics in one end-to-end stack. The frontier is a **composition of the best
pieces from each layer**: camera perception, explicit camera/world geometry, body
modeling, temporal priors, LiDAR or depth grounding, and physics/RL refinement.

**SOTA in 2026 = camera model + body model + temporal prior + scene geometry + physics layer.**

- LiDAR mainly improves geometry and grounding.
- RL mainly improves dynamics and inverse biomechanics.
- For a healthcare product, that modular combination is stronger than chasing a single
  "all-in-one" model.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Module-by-Module SOTA Methods](#module-by-module-sota-methods)
3. [Temporal Trends](#temporal-trends)
4. [Recommended 2026 Stack](#recommended-2026-stack)
5. [Block Diagram](#block-diagram)
6. [MSK / Rehab Product Recommendations](#msk--rehab-product-recommendations)
7. [Phased Build Plan](#phased-build-plan)
8. [References](#references)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     MODULAR SOTA STACK                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Layer 5: Physics / RL Refinement                               │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  PhysHMR · KinTwin · PhysPT · MuJoCo · OpenSim         │    │
│  │  → contact, torque, muscle activation, inverse dynamics │    │
│  └─────────────────────────────────────────────────────────┘    │
│                          ▲                                      │
│  Layer 4: Temporal Generative Prior                             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  RoHM · DPMesh · DiffCap · REWIND                      │    │
│  │  → occlusion repair, drift correction, ambiguity        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                          ▲                                      │
│  Layer 3: Body Model                                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  SMPL-X (mesh) + SKEL (biomechanics)                    │    │
│  │  → renderable mesh + clinically meaningful skeleton      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                          ▲                                      │
│  Layer 2: Geometry Stream (LiDAR / Depth)                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  LiveHPS · FreeCap · Point2Pose                         │    │
│  │  → metric scale, world coords, scene contact            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                          ▲                                      │
│  Layer 1: Perception Stream (RGB Camera)                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  4DHumans · MetricHMR · ProxyCap                        │    │
│  │  → monocular mesh recovery, 3D tracking                 │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module-by-Module SOTA Methods

### 1. Camera-First Monocular Motion Capture

| Method              | Contribution                                              | Reference |
|---------------------|-----------------------------------------------------------|-----------|
| **4DHumans**        | Strong baseline for monocular mesh recovery and 3D tracking | [arXiv:2305.20091](https://arxiv.org/abs/2305.20091) |
| **ProxyCap**        | Real-time world-space capture with physically plausible foot-ground contact | — |
| **Motions as Queries** | One-stage multi-person whole-body capture              | — |
| **MetricHMR / MetricHMSR** | Metric scale and global translation via camera rays and perspective geometry | [arXiv:2506.09919](https://arxiv.org/abs/2506.09919) |
| **BioPose + SKEL-based (HSMR, SKEL-CF)** | Biomechanical fidelity over visual plausibility | — |

### 2. LiDAR / Multimodal Grounding

| Method              | Contribution                                              | Reference |
|---------------------|-----------------------------------------------------------|-----------|
| **LiDARCap**        | Long-range single-LiDAR motion capture                    | [arXiv:2203.14698](https://arxiv.org/abs/2203.14698) |
| **LiDARHuman26M**   | Large-scale LiDAR human dataset                           | — |
| **LiveHPS**         | Scene-level human pose and shape from LiDAR               | [arXiv:2402.17171](https://arxiv.org/abs/2402.17171) |
| **FreeMotion**      | Scene-level estimation from LiDAR                         | — |
| **LiCamPose**       | Fused multi-view RGB with sparse point clouds             | — |
| **FreeCap**         | Single LiDAR + moving cameras, no fixed calibration       | — |
| **Point2Pose**      | Generative path for sequential point-cloud pose (WACV 2026) | — |

**Useful multimodal datasets:** RELI11D, AscendMotion (RGB + LiDAR + IMU supervision).

### 3. Body Model Layer

| Model               | Role                                                      | Reference |
|---------------------|-----------------------------------------------------------|-----------|
| **SMPL-X**          | Standard expressive mesh for body, hands, and face        | [arXiv:1904.05866](https://arxiv.org/abs/1904.05866) |
| **BlazePose GHUM Holistic** | Lightweight / on-device fallback                  | — |
| **SKEL**            | Biomechanically accurate skeleton re-rigging of SMPL      | — |
| **HSMR / SKEL-CF**  | Early estimators on the SKEL representation               | — |

### 4. Physics / RL / Biomechanics

| Method              | Contribution                                              | Reference |
|---------------------|-----------------------------------------------------------|-----------|
| **PhysPT**          | Force, contact, and Euler-Lagrange losses for physical plausibility | [arXiv:2404.04430](https://arxiv.org/abs/2404.04430) |
| **OSDCap**          | Learnable filtering and dynamics estimation                | — |
| **PhysDynPose**     | Scene geometry and physical constraints for moving cameras / non-flat scenes | — |
| **PhysHMR**         | Vision-conditioned control policy in simulator with expert distillation + RL | [arXiv:2510.02566](https://arxiv.org/abs/2510.02566) |
| **KinTwin**         | Imitation learning → torque-driven and muscle-driven biomechanical models | — |

**Simulation anchors:** MuJoCo (contact-rich physics), OpenSim (inverse dynamics / MSK).

---

## Temporal Trends

**Diffusion and masked motion priors** are now used to repair ambiguity, occlusion, and
temporal drift:

| Method        | Focus                                                        | Reference |
|---------------|--------------------------------------------------------------|-----------|
| **RoHM**      | Robust motion reconstruction under noisy monocular RGB(-D)   | [arXiv:2401.08570](https://arxiv.org/abs/2401.08570) |
| **DPMesh**    | Diffusion priors for occluded mesh recovery                  | — |
| **DiffCap**   | Fuses sparse IMUs + monocular camera with diffusion prior    | — |
| **REWIND**    | Real-time egocentric whole-body motion diffusion             | — |

Recent 2026 work explicitly attacks metric and temporal drift with **depth-guided
consistency terms**.

---

## Recommended 2026 Stack

Building one integrated system from the best available modules:

| # | Layer                          | Role                                                    | Key Models |
|---|--------------------------------|---------------------------------------------------------|------------|
| 1 | **RGB Perception Stream**      | Main perception; monocular video backbone               | 4DHumans, MetricHMR |
| 2 | **LiDAR Geometry Stream**      | Metric scale, world coordinates, occlusion, contact     | LiveHPS, FreeCap, Point2Pose |
| 3 | **Dual Body Representation**   | Renderable mesh + biomechanical skeleton                | SMPL-X + SKEL / OpenSim |
| 4 | **Temporal Generative Prior**  | Repair occlusion, front/back ambiguity, scale drift     | Diffusion-based (RoHM-style) |
| 5 | **Physics/RL Post-Processing** | Contact, torque, inverse dynamics, muscle activation    | PhysHMR, KinTwin, MuJoCo |

### Design Principles

1. **RGB video is the main perception stream.** Camera-only systems are already very
   strong and far easier to deploy than LiDAR-heavy rigs.
2. **LiDAR is the geometry stream, not the primary pose stream.** Use it to stabilize
   metric scale, world coordinates, long-range translation, occlusion, and scene contact.
3. **Predict two representations at once:** a renderable mesh (SMPL-X) and a
   biomechanics-facing representation (SKEL / OpenSim-compatible). This gives both
   UX-quality visualization and clinically meaningful kinematics.
4. **Add a temporal generative prior** over a short window to repair occlusion,
   front/back ambiguity, and scale drift.
5. **Use physics/RL after kinematics are roughly solved.** Contact consistency, ground
   penetration, torque estimation, inverse dynamics, and muscle activation.

---

## Block Diagram

```
 ┌──────────────┐       ┌────────────────┐
 │  RGB Camera  │       │   LiDAR Sensor │
 │  (monocular) │       │  (optional V2) │
 └──────┬───────┘       └───────┬────────┘
        │                       │
        ▼                       ▼
 ┌──────────────┐       ┌────────────────┐
 │  Monocular   │       │  Point Cloud   │
 │  Backbone    │       │  Encoder       │
 │  (MetricHMR) │       │  (Point2Pose)  │
 │              │       │                │
 │  Outputs:    │       │  Outputs:      │
 │  • 2D/3D pose│       │  • Depth map   │
 │  • Camera    │       │  • Scene mesh  │
 │    intrinsics│       │  • Metric scale│
 └──────┬───────┘       └───────┬────────┘
        │                       │
        └───────────┬───────────┘
                    ▼
         ┌──────────────────┐
         │  Fusion Module   │
         │                  │
         │  • Metric scale  │
         │    alignment     │
         │  • World-space   │
         │    registration  │
         │  • Occlusion     │
         │    handling      │
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │  Body Model      │
         │  Layer           │
         │                  │
         │  Dual output:    │
         │  ┌────────────┐  │
         │  │  SMPL-X    │  │  ← Renderable mesh (visualization)
         │  │  mesh      │  │
         │  └────────────┘  │
         │  ┌────────────┐  │
         │  │  SKEL /    │  │  ← Biomechanical skeleton (clinical)
         │  │  OpenSim   │  │
         │  └────────────┘  │
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │  Temporal Prior  │
         │  (Diffusion)     │
         │                  │
         │  Window: ~1-2s   │
         │  • Occlusion     │
         │    repair        │
         │  • Ambiguity     │
         │    resolution    │
         │  • Drift         │
         │    correction    │
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │  Physics / RL    │
         │  Refinement      │
         │  (V3 only)       │
         │                  │
         │  • PhysHMR       │
         │    control       │
         │    policy        │
         │  • KinTwin       │
         │    muscle model  │
         │  • MuJoCo sim    │
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │  OUTPUTS         │
         │                  │
         │  • Joint angles  │
         │  • ROM metrics   │
         │  • Gait params   │
         │  • Contact forces│
         │  • Muscle acts   │
         │  • 3D mesh viz   │
         └──────────────────┘

 Sensor Assumptions:
   V1: Single RGB camera (smartphone or webcam), known/estimated intrinsics
   V2: + 1× solid-state LiDAR (e.g., Livox Mid-360 or iPad Pro dToF)
   V3: Same sensors, + GPU for RL inference

 Latency Targets:
   V1: <100 ms per frame (10+ FPS real-time)
   V2: <150 ms per frame (sensor fusion overhead)
   V3: <200 ms per frame (physics sim adds ~50 ms)
```

---

## MSK / Rehab Product Recommendations

**Go camera-first, then add LiDAR only where it buys you something real.**

The strongest clinical evidence is still **monocular / smartphone-first** rather than
LiDAR-first:

- **Portable Biomechanics Laboratory** validated handheld smartphone video against
  ground-truth motion capture on 15+ hours of data, reporting mean joint-angle errors
  below 3° and ICCs above 0.9 for gait metrics.
  ([arXiv:2507.08268](https://arxiv.org/abs/2507.08268))
- A 2026 upper-extremity reachable-workspace study found strong agreement for a frontal
  single-camera setup.
- **SAM4Dcap** points toward an open-source "biomechanical twin" pipeline by combining
  monocular 4D mesh recovery with OpenSim.

---

## Phased Build Plan

### V1 — Camera-First Clinical Capture

| Component             | Choice                              | Rationale |
|-----------------------|-------------------------------------|-----------|
| **Sensor**            | Single RGB camera (smartphone/webcam) | Lowest deployment barrier |
| **Perception**        | MetricHMR or 4DHumans               | Strong monocular baseline with metric scale |
| **Body model**        | SMPL-X + SKEL                       | Dual rendering + biomechanics |
| **Analysis**          | OpenSim-style inverse kinematics    | ROM, gait, compensation scoring |
| **Latency**           | < 100 ms/frame                      | Real-time clinic feedback |

**Sufficient for:** ROM assessment, gait analysis, compensation scoring, and many clinic
workflows.

### V2 — Add LiDAR for Geometry

| Component             | Choice                              | Rationale |
|-----------------------|-------------------------------------|-----------|
| **Additional sensor** | 1× solid-state LiDAR               | Global translation, scene geometry |
| **Fusion**            | LiveHPS / FreeCap style             | Stabilize world-space coordinates |
| **Added value**       | Crowded scenes, foot contact, occlusion, moving-camera | Handles edge cases V1 cannot |
| **Latency**           | < 150 ms/frame                      | Minor overhead from fusion |

### V3 — Physics / RL for the Kinetic Twin

| Component             | Choice                              | Rationale |
|-----------------------|-------------------------------------|-----------|
| **RL policy**         | PhysHMR-style control policy        | Contact plausibility, ground penetration |
| **Muscle model**      | KinTwin imitation learning          | Joint moments, muscle activations |
| **Simulator**         | MuJoCo (contact-rich) + OpenSim (MSK) | Force and inverse dynamics |
| **Added value**       | Kinetic outputs: forces, torques, muscle activity | Clinical-grade dynamics |
| **Latency**           | < 200 ms/frame                      | Physics sim adds ~50 ms |

### Phased Training Plan

```
Phase 1 (V1):
  ├── Train MetricHMR backbone on COCO + H3.6M + 3DPW
  ├── Fine-tune SKEL regressor on biomechanical ground truth
  ├── Validate against Portable Biomechanics Lab benchmarks
  └── Target: <3° mean joint-angle error on gait

Phase 2 (V2):
  ├── Pre-train point cloud encoder on LiDARHuman26M
  ├── Train fusion module on RELI11D / AscendMotion (RGB + LiDAR + IMU)
  ├── Fine-tune on clinic-specific scenes
  └── Target: metric-scale world-space accuracy <5 cm RMS

Phase 3 (V3):
  ├── Train PhysHMR control policy in MuJoCo simulator
  ├── Distill to lightweight inference network
  ├── Train KinTwin muscle model on OpenSim ground truth
  └── Target: physically plausible contact + <10% torque error vs. force plates
```

---

## References

1. Goel, S., et al. "Humans in 4D: Reconstructing and Tracking Humans with Transformers." ICCV 2023. [arXiv:2305.20091](https://arxiv.org/abs/2305.20091)
2. Li, Y., et al. "LiDARCap: Long-range Marker-less 3D Human Motion Capture with LiDAR Point Clouds." 2022. [arXiv:2203.14698](https://arxiv.org/abs/2203.14698)
3. Pavlakos, G., et al. "Expressive Body Capture: 3D Hands, Face, and Body from a Single Image." CVPR 2019. [arXiv:1904.05866](https://arxiv.org/abs/1904.05866)
4. Gartner, E., et al. "PhysPT: Physics-aware Pretrained Transformer for Estimating Human Dynamics from Monocular Videos." 2024. [arXiv:2404.04430](https://arxiv.org/abs/2404.04430)
5. Fan, J., et al. "RoHM: Robust Human Motion Reconstruction via Diffusion." CVPR 2024. [arXiv:2401.08570](https://arxiv.org/abs/2401.08570)
6. Chen, Y., et al. "LiveHPS: LiDAR-based Scene-level Human Pose and Shape Estimation in Free Environment." 2024. [arXiv:2402.17171](https://arxiv.org/abs/2402.17171)
7. Luo, Z., et al. "PhysHMR: Learning Humanoid Control Policies from Vision for Physically Plausible Human Motion Reconstruction." 2025. [arXiv:2510.02566](https://arxiv.org/abs/2510.02566)
8. Uhlrich, S., et al. "Portable Biomechanics Laboratory: Clinically Accessible Movement Analysis from a Handheld Smartphone." 2025. [arXiv:2507.08268](https://arxiv.org/abs/2507.08268)
9. Dwivedi, S., et al. "MetricHMR: Metric Human Mesh Recovery from Monocular Images." 2025. [arXiv:2506.09919](https://arxiv.org/abs/2506.09919)
