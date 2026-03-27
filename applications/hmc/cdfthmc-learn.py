#!/usr/bin/env python3
#
# Authors: Christoph Lehner
#
import gpt as g
import sys, os
import numpy as np

rad = g.ad.reverse

beta = 2.95

U = g.load("cdrhmc_16_0.5625x3/ckpoint_lat.99")
even, odd = g.even_odd_projectors(U[0].grid)
full = g(even + odd)
none = g(0 * full)

description = [
    [(0.1, g.path().f(nu).f(mu).b(nu, 4).b(mu).f(nu, 3)) for nu in range(4) if mu != nu]
    for mu in range(4)
]

pt_e = [
    g.qcd.gauge.smear.parallel_transport(
        U,
        description,
        [even if i == j else full for i in range(4)],
        [odd if i == j else none for i in range(4)],
    )
    for j in range(4)
]

pt_o = [
    g.qcd.gauge.smear.parallel_transport(
        U,
        description,
        [odd if i == j else full for i in range(4)],
        [even if i == j else none for i in range(4)],
    )
    for j in range(4)
]

inv_pt_e = [x.inv() for x in pt_e]
inv_pt_o = [x.inv() for x in pt_o]

# undo smearing
for i in reversed(range(4)):
    U = inv_pt_e[i](U)
    U = inv_pt_o[i](U)


# learnable
rho = rad.node(rad.node(-0.3 + 0j))
description = [
    [(rho, g.path().f(nu).f(mu).b(nu, 4).b(mu).f(nu, 3)) for nu in range(4) if mu != nu]
    for mu in range(4)
]

nnU = [rad.node(rad.node(u)) for u in U]

n_pt_e = [
    g.qcd.gauge.smear.parallel_transport(
        U,
        description,
        [even if i == j else full for i in range(4)],
        [odd if i == j else none for i in range(4)],
    )
    for j in range(4)
]

n_pt_o = [
    g.qcd.gauge.smear.parallel_transport(
        U,
        description,
        [odd if i == j else full for i in range(4)],
        [even if i == j else none for i in range(4)],
    )
    for j in range(4)
]

a1 = g.qcd.gauge.action.differentiable_iwasaki(beta)
nnU0 = nnU
for i in range(4):
    nnU = n_pt_o[i](nnU)
    nnU = n_pt_e[i](nnU)

a1(nnU)()

c = sum(g.norm2(nnU0[mu].gradient) for mu in range(4))/4/full.grid.fsites/8/3

for epoch in range(10):
    cv = c()
    g.message(epoch, cv, rho.value.value, rho.value.gradient)

    rho.value.value -= 1e-2*rho.value.gradient
