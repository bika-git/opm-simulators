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

#ifndef OPM_NEURAL_CPR_FEATURES_HPP
#define OPM_NEURAL_CPR_FEATURES_HPP

#include <algorithm>
#include <array>
#include <cmath>

namespace Opm {

/// Solver-state features extracted once per Newton iteration, fed into NeuralCPRPolicy.
/// All eight slots are normalised to [0, 1] by toInputVector() before MLP inference.
struct NeuralCPRFeatures {
    double dt_days            = 1.0;   ///< Current timestep in days
    double dt_ratio           = 1.0;   ///< dt / dt_prev  (1 on first step)
    double nl_residual_norm   = 1.0;   ///< L2 norm of current linear rhs
    double nl_residual_reduce = 1.0;   ///< norm / norm at Newton iter 0 for this step
    double nl_iteration       = 0.0;   ///< Newton iteration index (0-based)
    double nnz_per_row        = 7.0;   ///< Average non-zeros per matrix row
    double diag_dominance     = 1.0;   ///< Mean |a_ii|_F / sum_{j!=i} |a_ij|_F
    double pad                = 0.0;   ///< Reserved, always 0

    std::array<float, 8> toInputVector() const
    {
        auto clamp01 = [](double v) { return float(std::clamp(v, 0.0, 1.0)); };
        return {
            float(std::log1p(dt_days) / 7.0),
            clamp01(dt_ratio / 10.0),
            // map log10 residual from [−14, 0] → [0, 1]
            float((std::log10(std::max(nl_residual_norm, 1e-14)) + 14.0) / 14.0),
            clamp01(nl_residual_reduce),
            float(std::clamp(nl_iteration, 0.0, 20.0) / 20.0),
            float(std::log1p(nnz_per_row) / 5.0),
            clamp01(diag_dominance / 4.0),
            float(pad)
        };
    }
};

} // namespace Opm

#endif // OPM_NEURAL_CPR_FEATURES_HPP
