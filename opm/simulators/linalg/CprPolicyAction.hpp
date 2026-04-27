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

#ifndef OPM_CPR_POLICY_ACTION_HPP
#define OPM_CPR_POLICY_ACTION_HPP

namespace Opm {

/// Pressure-decoupling weight computation method.
enum class CprWeightType {
    QuasiIMPES,         ///< O(n) diagonal weight extraction
    TrueIMPES,          ///< O(n × numEq³) local-solve weights
    TrueIMPESAnalytic   ///< Analytic TrueIMPES variant
};

/// Smoother type for CPR fine-level and coarse AMG level.
enum class CprSmoother {
    ParOverILU0,   ///< Parallel ILU(0) with overlap
    DILU,          ///< Decoupled ILU
    ILU0           ///< Sequential ILU(0) (coarse level only)
};

/// A discrete CPR solver configuration produced by NeuralCprPolicy::predict().
///
/// Covers three independent axes:
///   - Pressure-decoupling strategy   (weight_type)
///   - Whether to use CPRW well-weights (use_cprw)
///   - Fine-level smoother type        (fine_smoother)
///   - Coarse AMG smoother type        (coarse_smoother)
struct CprPolicyAction {
    CprWeightType weight_type     = CprWeightType::TrueIMPES;
    bool          use_cprw        = true;
    CprSmoother   fine_smoother   = CprSmoother::DILU;
    CprSmoother   coarse_smoother = CprSmoother::ILU0;

    bool operator==(const CprPolicyAction& o) const noexcept
    {
        return weight_type     == o.weight_type
            && use_cprw        == o.use_cprw
            && fine_smoother   == o.fine_smoother
            && coarse_smoother == o.coarse_smoother;
    }
    bool operator!=(const CprPolicyAction& o) const noexcept { return !(*this == o); }
};

} // namespace Opm

#endif // OPM_CPR_POLICY_ACTION_HPP
