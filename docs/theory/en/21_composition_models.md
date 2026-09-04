# 21. Composition Models: CrabNet, Roost, and GP Layers

A composition vector lies on a simplex:

```math
c_j\ge 0,\qquad \sum_j c_j=1.
```

Composition-only models predict properties from elemental identities and fractions without requiring crystal coordinates.

## Learned composition encoders

CrabNet and Roost learn representations from formulas or fractional compositions. Their embeddings can be used directly for prediction or as feature maps feeding a GP/DKL uncertainty layer.

## Log-ratio coordinates

CLR and ILR transforms can be useful when treating positive compositions as compositional data. Zero-valued components require special handling, so element-subset selection and ratio optimization are often best modeled separately.

## Process conditions

A practical surrogate may use both composition representation and process variables, `y=f(c,p)`. This is often more realistic than optimizing composition alone.

## bochan perspective

Composition encoders, GP layers, subset selection, and process optimization are treated as separate responsibilities that can be composed.