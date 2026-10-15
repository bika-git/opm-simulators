/*
  Copyright 2026 OPM Contributors.

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

#ifndef OPM_NEURAL_ABORT_POLICY_HPP
#define OPM_NEURAL_ABORT_POLICY_HPP

#include <opm/common/OpmLog/OpmLog.hpp>
#include <opm/ml/ml_model.hpp>

#include <array>
#include <cmath>
#include <exception>
#include <string>

namespace Opm {

/// Neural early-abort policy for timestep attempts.
///
/// Before each CPR-preconditioned linear solve it estimates the probability
/// that the current timestep attempt will end in a chop anyway.  Above the
/// threshold the solve's iteration budget is set to zero: the solve fails at
/// once and flow chops the timestep, instead of first spending the remaining
/// Newton iterations and a full maxiter failing solve.
///
/// The model (train_abort_policy.py) is an MLP in the binary format read by
/// Opm::ML::NNModel whose single output is the logit of that probability.  It
/// sees only what the linear solver knows; see ISTLSolver::abortFeatures().
class NeuralAbortPolicy {
public:
    static constexpr int kNumFeatures = 14;
    using Features = std::array<float, kNumFeatures>;

    NeuralAbortPolicy(const std::string& model_path, double threshold)
        : threshold_(threshold)
    {
        try {
            valid_ = model_.loadModel(model_path);
        } catch (const std::exception& e) {
            OpmLog::warning("NeuralAbortPolicy: failed to load model from '"
                            + model_path + "': " + e.what() + " — aborts disabled");
            valid_ = false;
        }
    }

    bool isLoaded() const { return valid_; }
    double threshold() const { return threshold_; }

    /// Probability that the current attempt will be chopped (0 without a model).
    double doomProbability(const Features& x) const
    {
        if (!valid_)
            return 0.0;
        Opm::ML::Tensor<float> in(kNumFeatures);
        for (int i = 0; i < kNumFeatures; ++i)
            in(i) = x[i];
        Opm::ML::Tensor<float> out;
        if (!model_.apply(in, out) || out.data_.empty())
            return 0.0;
        return 1.0 / (1.0 + std::exp(-static_cast<double>(out.data_[0])));
    }

private:
    mutable Opm::ML::NNModel<float> model_;
    bool valid_ = false;
    double threshold_;
};

} // namespace Opm

#endif // OPM_NEURAL_ABORT_POLICY_HPP
