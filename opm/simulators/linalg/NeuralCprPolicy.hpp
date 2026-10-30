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

#ifndef OPM_NEURAL_CPR_POLICY_HPP
#define OPM_NEURAL_CPR_POLICY_HPP

#include <opm/simulators/linalg/CprPolicyAction.hpp>
#include <opm/simulators/linalg/CprPolicyFeatures.hpp>
#include <opm/simulators/linalg/PropertyTree.hpp>
#include <opm/ml/ml_model.hpp>

#include <array>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Opm {

/// Neural CPR policy backed by the OPM Kerasify ML framework.
///
/// MLP inference when a valid Kerasify model file is loaded (isLoaded() ==
/// true), via Opm::ML::NNModel<float>.  Without a usable model, or when the
/// model cannot be evaluated, it makes no decision and flow keeps its own
/// solver configuration.
///
/// The model file is produced by the companion Python export script using
/// opm-common's opm.ml.ml_tools.kerasify.export_model().
///
/// MLP architecture: Dense(14→32, relu) → Dense(32→16, relu) → Dense(16→24, linear)
///
/// Output: 24 logits over the joint 3×2×2×2 action space.
/// Decode: argmax index i → dec=i/8, cprw=(i/4)%2, fine=(i/2)%2, coarse=i%2
class NeuralCprPolicy {
public:
    /// Construct.  If model_path is empty or the file cannot be loaded,
    /// isLoaded() is false and the policy makes no decisions.
    explicit NeuralCprPolicy(const std::string& model_path = "");

    ~NeuralCprPolicy() = default;

    NeuralCprPolicy(const NeuralCprPolicy&)            = delete;
    NeuralCprPolicy& operator=(const NeuralCprPolicy&) = delete;

    bool isLoaded() const { return valid_; }

    /// Forbid a label index (0–23) from ever being selected by predict().
    /// When the model's top logit falls on a forbidden label, the next highest
    /// non-forbidden label is returned instead.  Safe to call multiple times.
    void forbidLabel(int label)
    {
        if (label >= 0 && label < 24)
            forbidden_mask_ |= (1u << label);
    }

    /// Predict the best CPR configuration for the given features, or nothing
    /// when the model cannot be evaluated (then flow keeps its configuration).
    /// If \p out_confidence is non-null it receives the softmax probability of
    /// the top label (0–1), or 0 when there is no prediction.
    /// Forbidden labels (see forbidLabel()) are never returned.
    std::optional<CprPolicyAction> predict(const CprPolicyFeatures& feat,
                                           float* out_confidence = nullptr) const;

    /// Deterministic CPR schedule from the Newton iteration index and the
    /// residual reduction (no model): cheap quasi-IMPES/ILU0 on the first
    /// Newton step, quasi-IMPES/DILU mid-convergence, true-IMPES/DILU when
    /// Newton struggles.  This was the branch's silent default until the
    /// 2026-10-29 inertness fix; it is now explicit (OPM_NEURAL_CPR_RULE=1).
    static CprPolicyAction ruleBasedPredict(const CprPolicyFeatures& feat);

    /// Cost-model mode (models from train_cpr_cost.py exported with tau=1,
    /// whose outputs are -log(cost) per label).  Returns the cheapest allowed
    /// label if it is predicted to beat \p anchor by more than \p min_gain
    /// (in log-cost, 0.05 ≈ 5 %), otherwise \p anchor.  If \p out_gain is
    /// non-null it receives the predicted log-cost gain of the cheapest label
    /// over the anchor.  Without a loaded model the anchor is returned.
    int chooseAnchored(const CprPolicyFeatures& feat, int anchor,
                       float min_gain, float* out_gain = nullptr) const;

    /// AMG-parameter mode (train_cpr_amg_policy.py): -log cost scores for the
    /// prolongation damping levels kAmgDamping and the coarsening target levels
    /// kAmgCoarsenTarget, either one per combination (3 x 3 outputs, damping
    /// major) or additive per parameter (3 + 3).  Returns the (damping, target)
    /// level pair with the highest score if it beats the default pair (0, 0) by
    /// more than \p min_gain in log-cost, otherwise (0, 0).  \p out_gain
    /// receives that predicted gain.
    std::pair<int, int> chooseAmgAnchored(const CprPolicyFeatures& feat, float min_gain,
                                          float* out_gain = nullptr) const;
    static constexpr std::array<double, 3> kAmgDamping{1.0, 1.3, 1.6};
    static constexpr std::array<int, 3> kAmgCoarsenTarget{1200, 600, 300};

    bool isForbidden(int label) const
    {
        return label < 0 || label >= 24 || (forbidden_mask_ & (1u << label));
    }

    /// Decode a joint label 0–23 (dec*8 + cprw*4 + fine*2 + coarse).
    static CprPolicyAction actionFromLabel(int label);

    /// Build a complete PropertyTree mirroring setupCPRW() in
    /// setupPropertyTree.cpp, accepted by FlexibleSolver without modification.
    static PropertyTree configToPropertyTree(const CprPolicyAction& act,
                                             double tol     = 0.005,
                                             int    maxiter = 20);

    /// Patch only the 1–3 fields that changed; avoids a full 30-node ptree rebuild.
    /// Precondition: act.use_cprw == prev.use_cprw.
    static void updatePropertyTreeInPlace(const CprPolicyAction& act,
                                          const CprPolicyAction& prev,
                                          PropertyTree& prm);

    static std::string weightTypeStr(CprWeightType w);
    static std::string fineSmootherStr(CprSmoother s);
    static std::string coarseSmootherStr(CprSmoother s);

private:
    mutable Opm::ML::NNModel<float> model_;
    bool     valid_            = false;
    int      model_input_size_ = CprPolicyFeatures::kNumFeatures;
    uint32_t forbidden_mask_   = 0;  ///< bitmask of labels never to select

    /// Read the input feature count from the first Dense layer of a binary model file.
    /// Returns CprPolicyFeatures::kNumFeatures on any parse failure.
    static int readModelInputSize(const std::string& path);

    /// Run the MLP; forbidden labels are set to -inf.  False on failure.
    bool scores(const CprPolicyFeatures& feat, std::vector<float>& out) const;

    static CprPolicyAction decodeLogits(const std::vector<float>& logits);
};

} // namespace Opm

#endif // OPM_NEURAL_CPR_POLICY_HPP
