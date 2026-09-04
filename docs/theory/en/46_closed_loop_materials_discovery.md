# 46. Closed-Loop Materials Discovery

Closed-loop discovery repeatedly connects modeling, decision making, execution, and data ingestion:

```text
design space
 -> representation
 -> probabilistic surrogate
 -> acquisition
 -> candidate
 -> DFT / experiment
 -> validation
 -> model update
 -> repeat
```

A production loop needs machine-readable design-space constraints, provenance, failure handling, pending-evaluation tracking, and reproducible model/acquisition configuration.

Human approval can remain part of the loop; closed-loop does not require fully autonomous laboratory control. Batch and asynchronous execution require q-acquisition and pending-point handling. Multi-fidelity loops additionally choose the evaluation source. Robust loops optimize under process/input variation.

Monitoring should include best observed value or hypervolume, posterior calibration, candidate diversity, failure rate, consumed cost, and cycle time.

`bochan` fits naturally as the probabilistic decision platform between data/representation and external evaluators.