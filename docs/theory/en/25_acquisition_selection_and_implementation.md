# 25. Acquisition Selection and Implementation

Acquisition choice depends on objective type, observation noise, batch size, constraints, and the value of non-myopic information.

## Single-objective BO

EI/LogEI is a strong default for improvement-focused BO. NEI/LogNEI handles noisy observations. UCB exposes an explicit exploration parameter. PI is simple but can be overly exploitative. Thompson sampling is useful for scalable randomized selection. KG and multi-step lookahead value information beyond immediate improvement but are more expensive.

## Multi-objective BO

EHVI/NEHVI optimize expected hypervolume improvement. NParEGO scalarizes objectives and can be attractive when many objectives or simpler optimization are desired.

## Classification and ordinal outputs

Acquisitions should operate in the correct probability or utility space rather than blindly treating latent scores as regression targets.

## bochan perspective

The acquisition package is organized by response family and task: regression/binary/multiclass/ordinal/non-Gaussian crossed with BO/AL/LSE.