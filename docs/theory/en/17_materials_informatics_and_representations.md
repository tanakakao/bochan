# 17. Materials Informatics and Representations

Materials optimization spans several distinct objects: element sets, compositions, crystal structures, process conditions, material states, and measured properties. A useful abstraction is

```math
y=f(c,S,p)+\epsilon,
```

where `c` is composition, `S` is structure, and `p` is process information.

## Representation layers

Composition-only models encode elemental identities and fractions. Structure-aware models additionally use atomic species, coordinates, and lattice information. Process variables remain ordinary decision variables unless a model explicitly embeds them.

For crystal structures, invariance to translation and atom indexing and appropriate rotational invariance/equivariance are central. Graph and equivariant neural networks therefore provide natural structure representations.

## bochan perspective

`bochan` separates representation from probabilistic decision making. CrabNet/Roost-style encoders address composition representation; ALIGNN-style models address structure representation; GP/DKL layers provide posterior uncertainty; BO/AL acquisitions decide what to evaluate next.

The key design rule is to choose the representation for the information actually available at decision time rather than assuming composition, structure, and process are interchangeable.