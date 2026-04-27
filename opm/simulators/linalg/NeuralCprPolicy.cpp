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

#include <opm/simulators/linalg/NeuralCprPolicy.hpp>

#include <opm/common/OpmLog/OpmLog.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace Opm {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

NeuralCprPolicy::NeuralCprPolicy(const std::string& model_path)
{
    if (model_path.empty())
        return;

    // NNModel::loadModel uses OPM_ERROR_IF which throws on file-open failure.
    try {
        valid_ = model_.loadModel(model_path);
    } catch (const std::exception& e) {
        OpmLog::warning("NeuralCprPolicy: failed to load model from '"
                        + model_path + "': " + std::string(e.what())
                        + " — using rule-based fallback");
        valid_ = false;
    }
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

CprPolicyAction NeuralCprPolicy::predict(const CprPolicyFeatures& feat,
                                          float* out_confidence) const
{
    if (!valid_) {
        if (out_confidence) *out_confidence = 1.0f;
        return ruleBasedPredict(feat);
    }

    auto arr = feat.toArray();
    Opm::ML::Tensor<float> in(CprPolicyFeatures::kNumFeatures);
    for (int i = 0; i < CprPolicyFeatures::kNumFeatures; ++i)
        in(i) = arr[i];

    Opm::ML::Tensor<float> out;
    if (!model_.apply(in, out) || static_cast<int>(out.data_.size()) < 24) {
        if (out_confidence) *out_confidence = 1.0f;
        return ruleBasedPredict(feat);
    }

    if (out_confidence) {
        // Softmax over 24 logits, numerically stable.
        const auto* d  = out.data_.data();
        float max_l    = *std::max_element(d, d + 24);
        float sum      = 0.f;
        for (int i = 0; i < 24; ++i) sum += std::exp(d[i] - max_l);
        const int best = static_cast<int>(
            std::max_element(d, d + 24) - d);
        *out_confidence = std::exp(d[best] - max_l) / sum;
    }

    return decodeLogits(out.data_);
}

CprPolicyAction NeuralCprPolicy::decodeLogits(const std::vector<float>& logits)
{
    // Joint 3×2×2×2 = 24-way output.
    // Index i encodes: dec = i/8, cprw = (i/4)%2, fine = (i/2)%2, coarse = i%2
    const auto best = std::max_element(logits.begin(), logits.begin() + 24);
    const int  idx  = static_cast<int>(std::distance(logits.begin(), best));

    const int dec    = idx / 8;
    const int cprw   = (idx / 4) % 2;
    const int fine   = (idx / 2) % 2;
    const int coarse = idx % 2;

    CprPolicyAction act;
    switch (dec) {
    case 0:  act.weight_type = CprWeightType::QuasiIMPES;        break;
    case 1:  act.weight_type = CprWeightType::TrueIMPES;         break;
    default: act.weight_type = CprWeightType::TrueIMPESAnalytic; break;
    }
    act.use_cprw        = (cprw   == 1);
    act.fine_smoother   = (fine   == 0) ? CprSmoother::ParOverILU0 : CprSmoother::DILU;
    act.coarse_smoother = (coarse == 0) ? CprSmoother::ILU0        : CprSmoother::DILU;
    return act;
}

CprPolicyAction NeuralCprPolicy::ruleBasedPredict(const CprPolicyFeatures& feat)
{
    CprPolicyAction act;
    act.use_cprw        = true;
    act.coarse_smoother = CprSmoother::ILU0;

    const bool slowConv = (feat.nl_iteration > 0 && feat.nl_residual_reduce > 0.5);
    const bool manyIter = (feat.nl_iteration >= 4);

    if (slowConv || manyIter) {
        // Struggling to converge — use the most robust (expensive) config.
        act.weight_type   = CprWeightType::TrueIMPES;
        act.fine_smoother = CprSmoother::DILU;
    } else if (feat.nl_iteration == 0) {
        // First Newton step: prefer cheap weight computation and smoother setup.
        act.weight_type   = CprWeightType::QuasiIMPES;
        act.fine_smoother = CprSmoother::ParOverILU0;
    } else {
        // Mid-convergence: balanced choice.
        act.weight_type   = CprWeightType::QuasiIMPES;
        act.fine_smoother = CprSmoother::DILU;
    }
    return act;
}

// ---------------------------------------------------------------------------
// PropertyTree helpers
// ---------------------------------------------------------------------------

PropertyTree NeuralCprPolicy::configToPropertyTree(const CprPolicyAction& act,
                                                    double tol, int maxiter)
{
    using namespace std::string_literals;
    PropertyTree prm;

    prm.put("maxiter",   maxiter);
    prm.put("tol",       tol);
    prm.put("verbosity", 0);
    prm.put("solver",    "bicgstab"s);

    prm.put("preconditioner.type",             act.use_cprw ? "cprw"s : "cpr"s);
    prm.put("preconditioner.use_well_weights", act.use_cprw ? "true"s : "false"s);
    prm.put("preconditioner.add_wells",        act.use_cprw ? "true"s : "false"s);
    prm.put("preconditioner.weight_type",      weightTypeStr(act.weight_type));
    prm.put("preconditioner.pre_smooth",       0);
    prm.put("preconditioner.post_smooth",      1);
    prm.put("preconditioner.finesmoother.type",       fineSmootherStr(act.fine_smoother));
    prm.put("preconditioner.finesmoother.relaxation", 1.0);
    prm.put("preconditioner.verbosity",        0);
    prm.put("preconditioner.coarsesolver.maxiter",   1);
    prm.put("preconditioner.coarsesolver.tol",       1e-1);
    prm.put("preconditioner.coarsesolver.solver",    "loopsolver"s);
    prm.put("preconditioner.coarsesolver.verbosity", 0);

    const std::string amg = "preconditioner.coarsesolver.preconditioner.";
    prm.put(amg + "type",                "amg"s);
    prm.put(amg + "smoother",            coarseSmootherStr(act.coarse_smoother));
    prm.put(amg + "alpha",               0.333333333333);
    prm.put(amg + "relaxation",          1.0);
    prm.put(amg + "iterations",          1);
    prm.put(amg + "coarsenTarget",       1200);
    prm.put(amg + "pre_smooth",          1);
    prm.put(amg + "post_smooth",         1);
    prm.put(amg + "beta",                0.0);
    prm.put(amg + "verbosity",           0);
    prm.put(amg + "maxlevel",            15);
    prm.put(amg + "skip_isolated",       0);
    prm.put(amg + "accumulate",          1);
    prm.put(amg + "prolongationdamping", 1.0);
    prm.put(amg + "maxdistance",         2);
    prm.put(amg + "maxconnectivity",     15);
    prm.put(amg + "maxaggsize",          6);
    prm.put(amg + "minaggsize",          4);
    return prm;
}

void NeuralCprPolicy::updatePropertyTreeInPlace(const CprPolicyAction& act,
                                                 const CprPolicyAction& prev,
                                                 PropertyTree& prm)
{
    if (act.weight_type != prev.weight_type)
        prm.put("preconditioner.weight_type", weightTypeStr(act.weight_type));
    if (act.fine_smoother != prev.fine_smoother)
        prm.put("preconditioner.finesmoother.type", fineSmootherStr(act.fine_smoother));
    if (act.coarse_smoother != prev.coarse_smoother)
        prm.put("preconditioner.coarsesolver.preconditioner.smoother",
                coarseSmootherStr(act.coarse_smoother));
}

// ---------------------------------------------------------------------------
// String helpers
// ---------------------------------------------------------------------------

std::string NeuralCprPolicy::weightTypeStr(CprWeightType w)
{
    switch (w) {
    case CprWeightType::QuasiIMPES:        return "quasiimpes";
    case CprWeightType::TrueIMPES:         return "trueimpes";
    case CprWeightType::TrueIMPESAnalytic: return "trueimpesanalytic";
    }
    return "trueimpes";
}

std::string NeuralCprPolicy::fineSmootherStr(CprSmoother s)
{
    return (s == CprSmoother::DILU) ? "dilu" : "paroverilu0";
}

std::string NeuralCprPolicy::coarseSmootherStr(CprSmoother s)
{
    return (s == CprSmoother::DILU) ? "dilu" : "ilu0";
}

} // namespace Opm
