#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
import gpt as g
import sys


class defect_correcting:

    #
    # Less numerically stable (leads to suppression of critical low-mode space before inner_mat^{-1}):
    #
    # outer_mat = inner_mat (1 + eps defect)     (*)
    #
    # outer_mat^{-1} = (1 - eps defect + eps^2 defect^2 - eps^3 defect^3 + ...) inner_mat^{-1}   (geometric series)
    #
    # lhs = outer_mat^{-1} rhs
    #     = (1 - eps defect + eps^2 defect^2 - eps^3 defect^3 + ...) inner_mat^{-1} rhs
    #
    # Now defining lhs^{(0)} = inner_mat^{-1} rhs, we have
    #
    #     lhs = lhs^{(0)} - eps defect lhs^{(0)} + eps^2 defect^2 lhs^{(0)} + ...
    #
    # We can rearrange (*) to find
    #
    #     eps defect = inner_mat^{-1} outer_mat - 1
    #

    #
    # More numerically stable (implemented here):
    #
    # outer_mat =  (1 + eps defect) inner_mat     (*)
    #
    # outer_mat^{-1} = inner_mat^{-1} (1 - eps defect + eps^2 defect^2 - eps^3 defect^3 + ...)   (geometric series)
    #
    # lhs = outer_mat^{-1} rhs
    #     = inner_mat^{-1} (1 - eps defect + eps^2 defect^2 - eps^3 defect^3 + ...) rhs
    #
    # Now defining lhs^{(0)} = inner_mat^{-1} rhs, we have
    #
    #     lhs = lhs^{(0)} - inner_mat^{-1} eps defect rhs + ...
    #
    # We can rearrange (*) to find
    #
    #     eps defect = outer_mat inner_mat^{-1} - 1
    #
    # Therefore
    #
    #     lhs = lhs^{(0)} - inner_mat^{-1} (outer_mat inner_mat^{-1} - 1) rhs + ...
    #
    #     lhs = lhs^{(0)} - inner_mat^{-1} (outer_mat lhs^{(0)} - rhs) + ...
    #
    # Finally
    #
    #     lhs^{(0)} = inner_mat^{-1} rhs
    #     lhs^{(1)} = lhs^{(0)} - inner_mat^{-1} (outer_mat lhs^{(0)} - rhs)
    #     lhs^{(2)} = lhs^{(1)} - inner_mat^{-1} (outer_mat lhs^{(1)} - rhs)
    #

    @g.params_convention(eps=1e-15, maxiter=1000000)
    def __init__(self, inner_inverter, params):
        self.params = params
        self.eps = params["eps"]
        self.maxiter = params["maxiter"]
        self.history = None
        self.inner_inverter = inner_inverter

    def __call__(self, outer_mat):

        inner_inv_mat = self.inner_inverter(outer_mat)

        def inv(psi, src):

            # verbosity
            verbose = g.default.is_verbose("dci")
            t = g.timer("dci")
            t("setup")

            # inner source
            n = len(src)
            _s = [g.lattice(x) for x in src]

            # norm of source
            norm2_of_source = g.norm2(src)
            norm2_of_source = [
                g.norm2(outer_mat * psi[j])
                if norm2_of_source[j] == 0.0
                else norm2_of_source[j]
                for j in range(n)
            ]
            norm2_of_source = [
                1.0 if norm2_of_source[j] == 0.0 else norm2_of_source[j]
                for j in range(n)
            ]

            self.history = []
            for i in range(self.maxiter):

                t("outer matrix")
                for j in range(n):
                    _s[j] @= src[j] - outer_mat * psi[j]  # remaining src

                t("norm2")
                norm2_of_defect = g.norm2(_s)

                # true resid
                t("norm2")
                eps = max(
                    [(norm2_of_defect[j] / norm2_of_source[j]) ** 0.5 for j in range(n)]
                )
                self.history.append(eps)

                if verbose:
                    g.message("Defect-correcting inverter: res^2[ %d ] = %g" % (i, eps))

                if eps < self.eps:
                    if verbose:
                        t()
                        g.message(
                            "Defect-correcting inverter: converged in %d iterations, took %g s"
                            % (i + 1, t.total)
                        )
                        g.message(t)
                    break

                # normalize _s to avoid floating-point underflow in inner_inv_mat
                t("linear algebra")
                for j in range(n):
                    _s[j] /= norm2_of_source[j] ** 0.5

                # correction step
                t("inner inverter")
                _d = g.eval(inner_inv_mat * _s)

                t("linear algebra")
                for j in range(n):
                    psi[j] += _d[j] * norm2_of_source[j] ** 0.5

        otype, grid, cb = None, None, None
        if type(outer_mat) == g.matrix_operator:
            otype, grid, cb = outer_mat.otype, outer_mat.grid, outer_mat.cb

        return g.matrix_operator(
            mat=inv,
            inv_mat=outer_mat,
            otype=otype,
            accept_guess=(True, False),
            grid=grid,
            cb=cb,
            accept_list=True,
        )
