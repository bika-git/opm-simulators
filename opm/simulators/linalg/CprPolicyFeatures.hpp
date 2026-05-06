/*
  Copyright 2024 OPM Contributors.

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef OPM_CPR_POLICY_FEATURES_HPP
#define OPM_CPR_POLICY_FEATURES_HPP

#include <algorithm>
#include <array>
#include <cmath>

namespace Opm {

/// Fourteen scalar features describing the linear system and Newton state
/// at the moment ISTLSolver::prepare() is called.
///
/// toArray() normalises each value to approximately [0, 1] using log
/// transforms and clamping so no per-run standardisation is needed at
/// inference time.  The normalisation constants are chosen to cover the
/// dynamic ranges typical of OPM reservoir simulations.
struct CprPolicyFeatures {
    double dt_days              = 1.0;  ///< current timestep in days
    double dt_ratio             = 1.0;  ///< dt / prev_dt  (1.0 on the first step)
    double nl_residual_norm     = 1.0;  ///< L2 norm of the assembled right-hand side
    double nl_residual_reduce   = 1.0;  ///< norm / norm_at_nl_iter_0  (1.0 on iter 0)
    double nl_iteration         = 0.0;  ///< Newton iteration index (0-based)
    double nnz_per_row          = 7.0;  ///< average non-zeros per block row (cached)
    double diag_dominance       = 1.0;  ///< mean ||a_ii||_F / sum_{j≠i} ||a_ij||_F
    double prev_linsolver_iters = 0.0;  ///< Krylov iterations in the previous solve
    double prev_solve_failed    = 0.0;  ///< 1.0 if the previous linear solve failed
    double time_elapsed_frac    = 0.0;  ///< simulator.time() / simulator.endTime()
    double num_cells_log            = 3.0;  ///< log10(number of grid cells)
    double block_size               = 3.0;  ///< number of equations per cell (numEq)
    double well_density             = 0.0;  ///< num_wells / num_cells (coupling intensity)
    double nl_residual_trend        = 1.0;  ///< prev_norm / prev2_norm (>1=diverging)
    // Medium-impact features
    double num_phases               = 3.0;  ///< active phases: 1, 2, or 3
    double dt_cut_count             = 0.0;  ///< consecutive timestep cuts (resets on success)
    double prev2_linsolver_iters    = 0.0;  ///< Krylov iterations two solves ago (trend signal)
    double condition_number_estimate= 1.0;  ///< max_diag / min_diag over sampled rows
    double bhp_well_fraction        = 0.0;  ///< fraction of wells currently on BHP control

    static constexpr int kNumFeatures = 19;

    /// Return all features normalised to approximately [0, 1].
    std::array<float, kNumFeatures> toArray() const
    {
        auto c01 = [](double v) -> float {
            return static_cast<float>(std::clamp(v, 0.0, 1.0));
        };
        return {{
            // original 12
            static_cast<float>(std::log1p(dt_days) / 7.0),
            c01(dt_ratio / 10.0),
            static_cast<float>((std::log10(std::max(nl_residual_norm, 1e-14)) + 14.0) / 14.0),
            c01(nl_residual_reduce),
            c01(nl_iteration / 20.0),
            static_cast<float>(std::log1p(nnz_per_row) / 5.0),
            c01(diag_dominance / 4.0),
            c01(prev_linsolver_iters / 50.0),
            static_cast<float>(std::clamp(prev_solve_failed, 0.0, 1.0)),
            c01(time_elapsed_frac),
            c01(num_cells_log / 6.0),
            c01(block_size / 6.0),
            // highest-impact new (sweep5)
            c01(well_density * 200.0),                       // 0.5% density → 1.0
            static_cast<float>(                              // log: 0.01→0, 1→0.5, 100→1
                (std::log10(std::clamp(nl_residual_trend, 0.01, 100.0)) + 2.0) / 4.0),
            // medium-impact new
            c01(num_phases / 3.0),                           // 1/2/3 → 0.33/0.67/1.0
            c01(dt_cut_count / 5.0),                         // 5 cuts → 1.0
            c01(prev2_linsolver_iters / 50.0),               // same scale as prev_linsolver_iters
            static_cast<float>(                              // log: 1→0, 1e5→0.5, 1e10→1
                std::clamp(std::log10(std::max(condition_number_estimate, 1.0)) / 10.0,
                           0.0, 1.0)),
            c01(bhp_well_fraction)                           // already [0,1]
        }};
    }
};

} // namespace Opm

#endif // OPM_CPR_POLICY_FEATURES_HPP
